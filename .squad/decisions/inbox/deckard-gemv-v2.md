# Decision drop — GEMV-v2 MLP deepening no-go

**Author:** Deckard  
**Branch:** `squad/int4-gemv-wideload-v2`  
**Date:** 2026-08-15  
**Scope:** native CUDA int4 M=1 decode GEMV, glm-4-9b block-128, H200 GPU6.

## Baseline re-profile

Command shape: `profile_native --model /home/justinchu/glm-e2e-artifacts/glm-4-9b-int4-cuda --ep cuda --backend native --steady --tokens 160 --decode-skip 40 --warmups 1 --runs 3`.

Final clean HEAD (no v2 code kept): **183.49 tok/s** median (runs 182.75 / 183.49 / 183.53), greedy token stream stable. This is below the prior quiet-host 192 tok/s and still below the certified ORT comparator (~250 tok/s).

Nsight Compute on `matmul_nbits_gemv_f16_general_bs_wide` (3 launches, `--set full --graph-profiling node`) showed the current v1 wide kernel already at the same wall as before:

| metric | median |
|---|---:|
| kernel time | 42.88 us |
| DRAM bytes/s | 1.407 TB/s |
| DRAM read pct of peak | 28.07% |
| L1/TEX / LSU throughput | 72.04% |
| SM throughput | 71.54% |
| active warps | 65.53% |
| registers/thread | 40 |
| global load instructions | 657,408 |
| global-load sectors | 15,887,360 |
| global-load bytes/sector | 17.77 B |
| long-scoreboard stall | 1.95 warps/issue |
| math-pipe throttle | 3.26 warps/issue |

**Limiter found:** not pure DRAM bandwidth. The v1 wide-load path is now L1/LSU + SM-issue co-bound (~72% L1/TEX, ~71% SM) with only ~28% DRAM read peak. More MLP alone cannot reach ORT's 2.42 TB/s unless it also lowers the per-output activation/scales/L1 traffic or instruction pressure.

## Levers tried

| lever | result | decision |
|---|---:|---|
| Shared activation row in CTA dynamic shared memory (`..._wide_smem`, K*fp32 stage once/CTA) | 91.99 tok/s smoke; large regression | reverted. Staging/sync/shared-read cost outweighed removing redundant activation L1 reads. |
| Depth-4 software pipeline (`..._wide_d4`) | e2e noisy 186.41 tok/s vs same-session D2 185.58; ncu worse: 47.62 us, 1.266 TB/s, active warps 45.0%, regs 57 | reverted. Extra prefetch registers dropped occupancy and slowed the actual target kernel. |
| 512-thread / 16-column CTA for wide single-warp path | e2e noisy 187.24 tok/s; ncu worse: 44.06 us, 1.368 TB/s, L1 70.16%, same 40 regs | reverted. Same resident warps with fewer CTAs did not improve MLP; kernel slower. |
| lm_head / final MatMul backup check | `ONNX_GENAI_PROFILE_OPS=1` steady captured window showed MatMul ~0.016 ms in the measured decode report; MatMulNBits remains dominant | did not pursue cuBLASLt int4 lm_head; not the current wall. |

No v2 kernel change survived the measurement gate.

## Correctness / regression gates run

- f64 dequant→GEMM oracle: `cargo test -p onnx-runtime-ep-cuda --features cuda,gpu-tests --test matmul_nbits_marlin_numerics --quiet` → **7/7 pass**.
- qwen regression smoke: `/home/justinchu/shared-models/qwen2.5-14b-instruct-int4-zp-onnx`, native CUDA, 64 tokens, decode-skip 16 → **154.71 tok/s**, stable generated tokens; no code change kept.
- glm greedy tokens: clean HEAD generated-token stream remained stable across runs; no source change kept, so byte identity to current main is preserved by construction.

## ORT comparison

Fresh ORT graph-off fair artifact (`/home/justinchu/glm-e2e-artifacts/glm-4-9b-int4-cuda-ortfair`, `ONNX_GENAI_ORT_LIB=/home/justinchu/onnx-genai/.ort-cuda-1.27/root/lib/libonnxruntime.so.1.27.0`) measured **196.18 tok/s** but is not the certified fastcfg comparator.

Attempting fresh ORT CUDA-graph fastcfg with `ONNX_GENAI_CUDA_GRAPH=1` failed during warmup with `ORT error: the ort_value must contain a constructed tensor or sparse tensor`; the non-ortfair artifact fails ORT load on `GroupQueryAttention.rotary_embedding_dim`. Therefore I did not produce a valid fresh 250 tok/s ORT run from this harness; the standing certified comparator remains ~250.3 tok/s.

## Final status

**NO-GO / hard wall proven for this v2 pass.** The important number is **183.49 tok/s native glm base decode vs ~250.3 tok/s certified ORT** (0.73× ORT), byte-identical by construction because all losing code was reverted. The next plausible direction is not deeper prefetch; it must reduce L1/LSU work per output (activation/scales traffic or instruction mix) while keeping 40-reg occupancy, or obtain a working ORT fastcfg harness to revalidate the comparator before more kernel surgery.
