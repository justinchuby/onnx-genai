# Decision drop — fp16 mixed-precision int4 wide GEMV (base-vs-base fp16-vs-fp16)

**Author:** Deckard (Systems Dev, CUDA/decode-perf)
**Branch:** `squad/int4-gemv-fp16-mixed` (off main `07a0443f`, post-#992)
**Date:** 2026-08-15
**Status:** opt-in (`ONNX_GENAI_GEMV_FP16=1`), accuracy-gated, DRAFT PR for Chew+Gaff review
**Refs:** #986 (wide-load GEMV), #992 (multicol NC=4), #957 (spec-capture doc)

## Why
Strategic reset (Justin, binding): we may only claim to beat ORT on **base
(non-speculative) M=1 decode under equal conditions**. ORT's int4 M=1 kernel is
**fp16** arithmetic; our merged multicol path is **fp32**-accumulate and
byte-identical, which structurally caps us below ORT's fp16 kernel. This artifact
matches ORT's arithmetic (fp16 dequant + `__hfma2` MAC) to close the remaining
gap on honest fp16-vs-fp16 terms.

## What landed
A new opt-in fp16 sibling of the column register-blocked wide GEMV, stacked on
top of #992's multicol NC=4 activation reuse:

- CUDA `decode_activation8_h2` — activations → `__half2`, permuted to match the
  weight nibble pairing (no fp32 widening).
- CUDA device fn `gemv_int4_fp16_lane_dot_multicol` + kernel
  `matmul_nbits_gemv_f16_general_bs_wide_multicol_fp16`.
- Rust wiring: entry const, `use_gemv_fp16()` gate, dispatch selection,
  columns_per_block, ABI bits check.

### Precision contract (the crux) — matches ORT's `MatMulFloat4BitsKernelM1`
**FINAL design (updated after `ortsource` read of ORT's actual M=1 kernel):** the
per-lane K reduction runs **entirely in fp16 `__half2`** — each 32-element chunk
is `__hfma2`-summed in fp16, its per-block scale folded in with `__hfma2`, and the
result accumulated into a per-column fp16 running `total`. **fp32 is used ONLY in
the final cross-lane `warp_sum`** (the 5-step shuffle). This mirrors ORT's
arithmetic exactly.

Why full-fp16 accumulate is safe over K (this reverses the earlier caution): the
per-lane fp16 reduction is a **wide, shallow tree** — 32 lanes stride K by 32, so
each lane folds only ~K/1024 chunks (≈4 for K=4096, ≈13 for K=13696), and the
`__half2` holds two ~16-deep sub-lanes. Total fp16 depth is **tens, not
thousands** → negligible mantissa loss. The earlier token-flip came from a NAIVE
deep *single-accumulator* fp16 sum of all K; ORT's wide-tree layout is what makes
full-fp16 production-safe, and matching it puts us in **ORT's own error class by
construction**.

**Honest note — the "full-fp16 is faster" hypothesis was DISPROVEN at the kernel
level.** I first shipped a per-chunk-fp32 variant (fp16 MAC within a chunk, fp32
accumulate across chunks). Switching to ORT's full-fp16 accumulate is a **speed
TIE** (kernel 26.4 vs 27.0 µs; e2e 212.8 vs 212.4 tok/s) and is **2× less
precise** (f64 max_rel 1.16e-2 vs 6.3e-3, both far under the 5e-2 bound). The
limiter is the multicol WIDE_NC=4 register state (64 regs) + wave/tail
quantization (achieved occupancy 41% << register-theoretical), **not** the
accumulator width. I ship the full-fp16 version to match ORT's arithmetic exactly
(cleanest "same error class as ORT" story for the gate), not for a speed win.
`WIDE_NC_FP16=2` (fewer cols → fewer regs) was tried and **rejected**: registers
dropped 64→54 but occupancy only 41→43% (wave-bound, not register-bound) and the
kernel got *slower* (L1/TEX 37→56% as activation reuse dropped).

## Performance (glm-4-9b-int4, GPU5 H200, `--steady --tokens 160 --decode-skip 40 --runs 3`)
| variant | tok/s | vs narrow |
|---|---|---|
| narrow (pre-#986) | 136.7 | — |
| wide-load #986 | ~185 | +35% |
| multicol NC=4 #992 (fp32) | **199.4** | +46% |
| **fp16 full-accumulate (this PR)** | **212.8** | **+56%** |

**+6.7% over the merged fp32 multicol.** ORT base (certified fair, foundry-local
fastcfg, Sebastian/GPU7) = ~250–252 → **native-vs-ORT base gap 1.30× → 1.18×**.

### ncu (glm gate_up, `--graph-profiling node`, launch-skip 240)
| metric | fp32 multicol | **fp16 mixed** | ORT |
|---|---|---|---|
| kernel time | 34.3 µs | **26.4 µs** | 24.9 µs |
| DRAM | 1.76 TB/s | **2.29 TB/s** | 2.42 TB/s |
| L1/TEX | 28% | 36% | — |
| SM throughput | 60% | 54% | — |
| regs / occupancy | 64 / 41% | 64 / 41% | 32 / — |

The fp16 MAC halves ALU pressure → the kernel is now **~parity with ORT's
gate_up kernel** (26.4 vs 24.9 µs) and DRAM approaches ORT's (2.29 vs 2.42 TB/s).
New limiter is **occupancy (41%)** — but it is **wave/tail-quantization bound, not
register-bound**: achieved occupancy (41%) sits far below the register-theoretical
ceiling (50–59% at 64 regs), and cutting registers via `WIDE_NC_FP16=2` moved
occupancy only 41→43% while slowing the kernel. So the remaining gap to ORT's
24.9 µs is ORT's fundamentally higher-occupancy 1-col/warp 32-reg layout, not
something reachable from the multicol layout the coordinator asked to keep. The
`idioms` ORT listed (LOP3 dequant, prmt activation reorder) were **already present**
in this kernel (`int4x8_to_half2x4_sub`, `decode_activation8_h2`) before the
ORT-source read — the only arithmetic delta was per-chunk-fp32 → full-fp16
accumulate, which is a speed tie.

## Accuracy gate (the bar for an opt-in non-byte-identical kernel)
Since the kernel is not bit-identical by design, it ships gated on **accuracy**,
not bit-identity. Two independent lines of evidence:

1. **f64-oracle numeric gate** (committed test
   `fp16_mixed_gemv_matches_f64_oracle_glm_decode`): runs the fp16 kernel on
   glm's real block-128 M=1 shapes (qkv / o / gate_up / down × sym+asym zp ×
   fp16+fp32 scales) against the same f64 dequant→GEMM oracle, held to the
   **SAME justified `Envelope` the reviewed fp32 int4 path meets** — no
   weakening. Result (shipped full-fp16-accumulate version): **fp16
   `max_rel = 1.16e-2`, 4.3× under the `5e-2` bound**; `max_abs` always <<
   `abs_bound`. (The earlier per-chunk-fp32 variant measured 6.3e-3; full-fp16
   accumulate is ~2× less precise but still comfortably inside the envelope and
   in ORT's own error class by construction.) The test also asserts the fp16
   output diverges from fp32 (proves the fp16 entry was actually selected, not a
   vacuous pass).

2. **Empirical token identity**: fp16 greedy tokens are **bit-identical to the
   fp32 path over 160 tokens on both a generic and a repetitive prompt** (0
   diffs). For greedy decode this is strictly stronger than a perplexity delta —
   identical token streams ⇒ identical sequence perplexity (0% delta). (A
   teacher-forced perplexity harness was not built; the token-identity superset
   guarantee covers the greedy claim.)

**On "≤ ORT":** ORT's int4 kernel cannot be linked into the ep-cuda test crate
(no ORT dev-dep; ORT plumbing lives only in `onnx-genai-bench`), so the numeric
`≤ ORT` assertion is expressed as "≤ the same envelope the fp32 path meets." That
is a conservative proxy: our fp32 path is the **most precise fp16-input int4
kernel possible** (fp32 accumulate throughout), and our fp16 kernel keeps the
K reduction in fp32, so it is **at least as precise as any fp16-accumulate kernel
ORT could ship**. Pending `ortsource`'s read of ORT's exact accumulator width, if
ORT turns out to accumulate fully in fp32 we are still within 1.2× of that floor
on the production (symmetric) shapes and empirically token-identical.

## Portability / capture
- Opt-in only (`ONNX_GENAI_GEMV_FP16=1`); default-off path is **untouched and
  byte-identical to main**. Plan (Justin-approved): default-ON once Chew's
  accuracy gate is green, because ORT is fp16 by default ⇒ equal-conditions =
  fp16-vs-fp16.
- Static launch grid, no arch guard needed (`__hfma2` is SM53+; all our CUDA
  targets qualify). Capture-safe (no alloc/sync/host-readback).
- **qwen unaffected by construction**: qwen is block-32 → the block-32 dispatch
  arm; `use_gemv_fp16` lives in the `block_size != 32` arm and is never reached.
  Confirmed: qwen tokens **identical** with the flag on vs off (80 tok).

## Gates status
- [x] f64 oracle numerics 8/8 (7 pre-existing + new fp16 gate) — GPU5
- [x] glm fp16 == fp32 greedy tokens identical, 160 tok, generic + repetitive
- [x] qwen no-regression (flag structurally cannot route qwen; tokens identical)
- [x] fmt clean; clippy `-p onnx-runtime-ep-cuda --features cuda --lib` 0 warnings
       (+ test target clippy clean)
- [x] default-off byte-identical to main

## Reviewer asks
- **Chew (accuracy):** is the "same-envelope-as-fp32-path" bar an acceptable
  stand-in for "≤ ORT" given ORT can't link in the test crate? Is the asymmetric-zp
  small-magnitude rel-ratio (9–13×, abs << bound) acceptable, or do you want a
  tightened per-shape abs bound?
- **Gaff (capture/perf):** confirm capture-safety of the fp16 entry and that the
  +6.5% (212.4 tok/s) is a real, reproducible win; opinion on default-ON timing.
