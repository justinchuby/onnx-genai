# Batty — CUDA Upstream Audit: Both Candidates Dead

**Date:** 2026-08-11
**Author:** Batty (Engine Dev, CUDA & Perf pod)
**Requested by:** @justinchuby
**Context:** PR #763 ranked two CUDA candidates for upstream contribution.

## Decision

**Neither CUDA candidate survives the audit. No portable gap was found.**

Both candidates — MatMulNBits int4 block-128 GEMV and QMoE parallel routing — are
already fully covered by upstream `main` or by in-flight Microsoft work. This mirrors
the CPU pilot outcome where 2/2 candidates died on inspection.

## Candidate 1: MatMulNBits int4 block-128 GEMV — DEAD

**Claimed gap:** Upstream's int4 GEMV doesn't handle `block_size=128`, giving us
+36–60% on 0.5B–1.5B models. Cited ORT issue #23004.

**Evidence that upstream already covers this:**

- `matmul_4bits_m1_impl.cuh:152` — explicit `block_size == 128` template instantiation
  in the M=1 GEMV kernel, with 16/32/64/128 all supported.
- `matmul_nbits.cc:76` — `CheckFpAIntBEligibility()` accepts
  `block_size == 32 || block_size == 64 || block_size == 128`.
- `fpA_intB_gemv/fpA_intB_gemv.h:36` — The CUTLASS-derived GEMV has a `groupsize`
  parameter that handles block-128 natively for both FP16 and BF16 inputs.
- `fpA_intB_gemv/dispatcher_fp16_int4.cu` and `dispatcher_bf16_int4.cu` — dispatchers
  instantiate the groupwise int4 GEMV path.

**Issue #23004 mismatch:** The issue is about *CPU* MatMulNBits performance (int4 vs
int8 on x86/ARM), not CUDA GEMV. It does not identify a CUDA gap.

**Verdict:** No gap. The +36–60% numbers measured *our* Rust runtime vs *our* baseline,
not upstream. Upstream's CUDA kernel already covers block-128 int4 GEMV.

## Candidate 2: QMoE Parallel Routing — DEAD

**Claimed gap:** Block-cooperative top-k replacing serial routing, +30% MoE decode.
Cited ORT issue #28987.

**Evidence that upstream already covers or is actively working on this:**

- PR #28980 "[CUDA] Optimize QMoE SoftmaxTopK router for small-batch decode" —
  **already closed/merged**.
- Issue #28987 is an active Microsoft-led tracking issue for Qwen3.6-35B-A3B
  throughput optimization, listing **8+ PRs** touching QMoE kernels:
  #28980, #28985, #28986, #29028, #29038, #29013, #29824, #29818.
- `qmoe_kernels.cu:30-34` — Already uses warp-cooperative reductions
  (`WarpReduceMax`, `WarpReduceSum`, `SafeInvSum`) from shared `topk_warp_sort.cuh`.
- `qmoe_kernels.h` shows extensive API surface including batched FP4, NVFP4,
  FP8 dequantization, scale interleaving, and SM90 TMA packing — far beyond
  what was in scope for our contribution.

**Verdict:** Duplicate of in-flight Microsoft work. Opening a PR here would waste
both our time and a reviewer's.

## Next-Best Options Evaluated

| Option | Why not viable |
|--------|---------------|
| CUDA `accuracy_level=4` (int8 activation quant) | Not implemented on CUDA EP, but tensor cores via fpA_intB already beat DP4A for bandwidth-bound GEMV. Not a real perf gap. |
| GGUF native block formats (IQ1S/IQ2XXS/MXFP4) | Custom op (`BlockQuantizedMatMul`) in our runtime only. Not in ORT's op spec; not upstreamable. |
| QMoE expert grouping for prefill | Microsoft actively iterating on MoE prefill routing (see `moe_quantization.cc` with FP4 prefill min tokens, SM80 grouped GEMM, SM90 TMA paths). Would duplicate. |
| Graph capture / VMM weight paging / tiered KV | Entangled with our memory governor and scheduler. Not portable — established by Batty in prior analysis. |

## Conclusion

This is a legitimate "no viable gap" outcome. The upstream CUDA EP is mature, well-staffed,
and actively iterated on by Microsoft engineers. Our biggest CUDA advantages are runtime-level
(graph capture, VMM, tiered KV) which are architecturally not portable.

**This is NOT ready for a draft upstream PR.** There is nothing to PR.

## What Pris Should Do

No CPU-reference harness is needed for this audit since no kernel was implemented.
Pris should continue with the existing test harness work (PR #768 blocker).

## What Remains GPU-Blocked

All original perf claims remain unmeasured against upstream code. The +36–60% and +30%
numbers measured our Rust runtime, not any upstream contribution. Issue #768 (GPU
validation environment) remains the blocker for any future CUDA upstream work.
