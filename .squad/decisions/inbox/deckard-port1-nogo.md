# Decision: PORT-1 (offline bias+interleave) is a measured NO-GO on the multicol asymmetric-zp GEMV

**Author:** Deckard (Systems Dev, CUDA/decode-perf)
**Date:** 2026-08-15
**Branch:** `squad/int4-gemv-port1-interleave` (stacked on `squad/int4-gemv-fp16-mixed` / #996)
**Status:** NO-GO — mechanism measured null with ncu BEFORE building the offline-repack surface.

## TL;DR
PORT-1 (bake the signed→unsigned +8 bias and the odd/even nibble interleave into
the stored weight layout so the runtime dequant drops ORT's `prmt.b32`) was ranked
as an occupancy lever: *cheaper dequant → fewer registers → higher occupancy →
faster*. **On our multicol fp16 GEMV that mechanism does not exist.** A direct ncu
probe of the exact op PORT-1 removes shows **0 registers freed, 0 occupancy
change, 0 kernel-time change.** Building the full offline-repack + versioned-layout
+ loader surface would deliver ~0 gain, so I stopped at the measurement rather than
spend days on a model-artifact change with no payoff.

## What PORT-1 actually removes on THIS kernel
The plan's PORT-1 win was derived from ORT's 1-column-per-warp symmetric kernel.
Two properties of our kernel collapse the win:

1. **Asymmetric zero-points (glm & qwen both use zp).** Our `int4x8_to_half2x4_sub`
   is already LOP3-based (4× `lop3` + 2× `sub.f16x2` + 2× `fma.f16x2`) **plus an
   irreducible 4× per-block zp-subtract**. PORT-1's `FastInterleavedAndBiasedNumericArrayConverter`
   only saves the 2× `sub.f16x2` that finalize a *symmetric* (+8-only) dequant. With
   asymmetric zp the offline +8 bias just shifts into the stored zp (`(w+8)-(zp+8)`)
   — same op count. **The weight-dequant instruction count does not drop.**
2. **The activation `prmt` is amortized across WIDE_NC=4 columns AND register-transient.**
   The only thing PORT-1 removes for the asymmetric case is `decode_activation8_h2`'s
   4× `prmt.b32`. But the activation sub-word is decoded ONCE and reused across all 4
   columns (the multicol L1 win), so it is ~1 prmt-equivalent per column per 8 weights
   (~2% of inner-loop ops), and the prmt is register-to-register → frees no registers.

## The measurement (ncu, glm gate_up matrix, GPU6, fresh idle, --graph-profiling node)
I added a `template<bool NATURAL>` to the fp16 multicol device fn: `NATURAL=true`
swaps `decode_activation8_h2` (4× prmt) for a prmt-free `decode_activation8_h2_natural`
(assumes offline-interleaved weights → activation maps straight to 4 `half2`). This
isolates *exactly* the op PORT-1 removes. `<false>` is byte-identical codegen to the
shipping #996 kernel (f64 oracle 8/8 still passes; registers/time identical).

| Metric | Baseline fp16 (#996) | prmt-free (PORT-1 dequant) | Δ |
|---|---|---|---|
| Registers / thread | 64 | 64 | **0** |
| Achieved occupancy | 41.17 % | 40.92 % | ~0 (noise) |
| Kernel time | 26.69 µs | 26.53 µs | −0.6 % (noise) |
| DRAM throughput | 2.267 TB/s | 2.272 TB/s | ~0 |

Probe entry gated behind `ONNX_GENAI_GEMV_FP16_NATURAL=1` (default OFF). Its output is
intentionally wrong vs the current non-interleaved weights — it exists ONLY to read
`launch__registers_per_thread`/occupancy (static compile properties, valid regardless
of data). Kept in the branch as reproducible evidence, never a correctness path.

## Why this is the third convergent "occupancy is the wall, and it doesn't move"
- register depth-4 prefetch: regs 40→57, occupancy 65.5%→45% → slower (reverted).
- cp.async double-buffer (#1000): hid Long-Scoreboard 2.35→1.88 but regs 64→71,
  occupancy 41.6%→33.3% → −4.2% (documented no-go).
- **PORT-1 (this): removing the only reducible dequant op frees 0 regs, 0 occupancy.**

The 64-register footprint is structural to WIDE_NC=4 (`w[4]` uint4 = 16 regs,
`col_kb[4]` long = 8, plus total/acc/scale2/sub2 per column). Dequant temporaries are
not the constraint. Occupancy will not rise without dropping WIDE_NC — which loses the
multicol L1 win that got us here.

## Bottom line for the base-vs-ORT goal
Our fp16 multicol GEMV is already at **26.5 µs / 2.27 TB/s (~95% of ORT's 2.42)** and
per-kernel near-parity with ORT. The remaining e2e gap to ORT ~250 base is dominated by
OTHER decode ops (lm_head — owned by deckard-4's cuBLASLt work), NOT this GEMV. Further
micro-opt of this kernel (cp.async, PORT-1) hits the same occupancy wall. Recommendation:
**land #996 fp16 (+6.7%), let deckard-4's lm_head land, and re-measure the base-vs-ORT
gap before spending more on this GEMV.** PORT-1's offline-repack surface is not worth
building for this kernel.
