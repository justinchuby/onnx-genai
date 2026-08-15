# Decision drop — GEMV v2: column register-blocked int4 wide GEMV (glm 184→197.6, byte-identical)

**Author:** Deckard (Systems Dev, CUDA/decode-perf) — sole driver, `matmul_nbits.rs`
**Branch:** `squad/int4-gemv-wideload-v2` (continuation of merged #986; refs #986)
**Date:** 2026-08-15  **GPU:** H200 GPU6 (verified 0 MiB / 0% idle before every run)
**Scope:** native CUDA int4 M=1 decode GEMV, glm-4-9b block-128; qwen block-32 as no-regression control.

## Headline
`ncu` proved the #986 wide-load GEMV is **NOT DRAM/load-MLP bound** (DRAM only ~28% / 1.40 TB/s) but
**L1/TEX-throughput bound (~72%)** — all 8 warps in a CTA redundantly re-read the full-K activation
row, so activation L1 traffic ≈ 4× the weight DRAM traffic. The original "deepen the load pipeline to
2.42 TB/s" premise was therefore **wrong** and is abandoned.

The winning lever is **column register-blocking**: each warp now emits `WIDE_NC=4` output columns,
decoding each 8-element activation sub-word to fp32 **once** and reusing it across the 4 columns
(~4× less activation L1 traffic). The 4 independent 128-bit weight loads per chunk supply the
memory-level parallelism — no software pipeline, no smem staging.

**glm base decode (fresh, GPU6, --steady --tokens 160 --decode-skip 40 --runs 3):**

| variant | tok/s |
|---|---:|
| narrow (`ONNX_GENAI_GEMV_WIDELOAD=0`) | 136.31 |
| wide single-col (#986) | 184.01 |
| **multicol NC=4 (v2, default)** | **197.62**  (+7.4% over #986, +45% over narrow) |

## ncu delta — gate_up GEMV, glm (fresh GPU6, `--set full`/`--metrics`, `--graph-profiling node`)

| metric | wide single-col (#986) | **multicol NC=4** |
|---|---:|---:|
| kernel time | 43.20 µs | **34.34 µs** |
| DRAM throughput | 1.40 TB/s | **1.76 TB/s** |
| **L1/TEX throughput** | **71.9 %** | **27.9 %**  ← limiter collapsed |
| SM throughput | 71.4 % | 60.5 % |
| registers/thread | 40 | 64 |
| active warps (occupancy) | 65.5 % | ~41 % |
| grid | 3424 | 856 (÷4, NC=4) |

L1/TEX (the real limiter) collapses 72%→28%; DRAM rises; kernel shrinks 43→34 µs. The new co-limiter
is SM/occupancy (60.5% SM at 64 regs / 41% occ) on the fp32 dequant ALU — see wall below.

## NC sweep (byte-identical at every NC)
NC=2 →196, NC=3 →195, **NC=4 →197.6 (best)**, NC=6 →182, NC=8 →182 (register cliff).
`__launch_bounds__(256, N)` register-capping regressed (local-mem spills): NC=5→187, NC=6→162.
**NC=4 is optimal**; the natural 64-reg / 41%-occ config beats any register cap.

## Correctness / regression gates (all PASS, fresh GPU6)
- **glm greedy BYTE-IDENTICAL** multicol vs narrow — full 160-token streams, **0 diffs**, on BOTH a
  generic prompt (Industrial Revolution) AND a repetitive prompt (ha-ha loop).
- **f64 dequant→GEMM oracle** `matmul_nbits_marlin_numerics` → **7/7 pass**.
- **qwen no-regression** (block-32, control): narrow 148.92 vs default 151.26 = within noise;
  tokens **byte-identical**. qwen never routes through `general_bs` (block!=32 arm), so multicol
  cannot touch it — unchanged by construction.
- **default-on, portable** (no SM arch guard — pure fp32 load-scheduling, byte-identical), **capture-safe**
  (static grid 856, no alloc/sync/host-readback). Opt-out `ONNX_GENAI_GEMV_WIDE_MULTICOL=0`.
- fmt clean; clippy `-p onnx-runtime-ep-cuda --features cuda --lib -D warnings` clean.

## Byte-identity mechanism (why it is bit-exact vs #986/narrow)
Per-column accumulation keeps the exact sequence `values[c] += scale*dot(w.x); += scale*dot(w.y); ...`
(4 separate fp32 adds, NOT `scale*(d0+d1+d2+d3)`). `decode_activation8` produces the fp32 activations
in the exact order `dot_int4x8_f16_sub` consumes; `dot_int4x8_f16_sub_act` does the identical 8-term
fp32 dot. Sub-word loop order 0→3 preserves ascending-K. Each column is therefore bit-for-bit equal to
#986's single-col wide → token-identical to narrow/main.

## base-vs-ORT (the honest headline)
- Native base decode arc: 136 (narrow) → 184 (#986) → **197.6 (v2)**.
- Certified ORT base (foundry-local glm-4-9b-fastcfg, CUDA-graph, ORT 1.27): **~250–252 tok/s** (Sebastian, GPU7).
  Gap 1.84× → 1.36× → **1.27×**.
- In-harness `--backend ort` on the same GPU6 measured **192 tok/s** (fastcfg) / 191.6 (ortfair) — but that
  path runs ORT WITHOUT CUDA graph, so it under-measures ORT and is NOT the fair comparator. ORT+CUDA-graph
  through `profile_native` fails at warmup (`ort_value must contain a constructed tensor` — a harness bug,
  reproduced independently; not our kernel). The fair bar stays the ~250 foundry number.

## STRUCTURAL WALL (strategic finding — needs a product ruling)
We did **not** reach ORT's ~250 base, and cannot while **byte-identical to the fp32 narrow path**. At NC=4 we
are now Compute-SM(60%)/occupancy(41%)-bound on the **fp32 dequant ALU**. ORT's `MatMulFloatInt4Kernel` gets
its speed from **fp16 dequant-math** (half the ALU). A pure-fp16 (half2 foldscale) MAC was measured earlier at
~204 tok/s BUT **flips one greedy token** on glm-generic → fails the byte-identical gate. So closing the last
~1.27× on the GEMV requires a ruling: **is byte-identical-to-narrow a hard requirement, or is oracle-gated
token-parity (like split-K #978) acceptable for the base GEMV?** If the latter, fp16-dequant math is the path
to ~250 base parity; if the former, 197.6 is the byte-identical ceiling and further base gains must come from
non-GEMV ops (e.g. fp16 lm_head).

## Rejected alternatives (measured, reverted — search was exhaustive)
| lever | result | verdict |
|---|---|---|
| Shared-activation smem (stage K*fp32 once/CTA, `__syncthreads`) | 91.99 tok/s (2× regression) | NO-GO — staging barrier serializes the load ahead of compute in a latency-bound kernel |
| Depth-4 software pipeline | e2e 186.41 vs D2 185.58 (noise); ncu worse 47.6 µs / 1.27 TB/s / 57 regs / 45% occ | NO-GO — extra prefetch registers kill occupancy, slow the kernel |
| 512-thread / 16-col CTA | ncu 44.06 µs / 1.368 TB/s (worse) | NO-GO — same resident warps, fewer CTAs, no MLP gain |
| MLP-deepening (original v2 premise) | — | ABANDONED — limiter is L1/TEX, not DRAM |
| fp16 dequant-math MAC | ~204 tok/s but flips a greedy token | BLOCKED on byte-identity — needs the ruling above |
| lm_head cuBLASLt | ~5.7% of decode, ~+2% projected; cuBLAS not bit-identical to hand GEMV | DEFERRED — separate scoping + parity ruling |

## Files
- `crates/onnx-runtime-ep-cuda/src/kernels/matmul_nbits.rs` — ONLY source changed. New CUDA helpers
  `decode_activation8` / `dot_int4x8_f16_sub_act`, device fn `gemv_int4_wide_lane_dot_multicol`
  (`#define WIDE_NC 4`), kernel `matmul_nbits_gemv_f16_general_bs_wide_multicol`; Rust entry const +
  `GEMV_F16_WIDE_MULTICOL_NC=4`, gate `use_gemv_wide_multicol()`, dispatch + `columns_per_block` + ABI wiring.

## Final status
**GO — byte-identical +7.4% base-decode win banked** (glm 197.6 tok/s, gap → 1.27× ORT). The remaining gap to
250 is a byte-identity-vs-fp16 product decision, not a kernel dead-end.
