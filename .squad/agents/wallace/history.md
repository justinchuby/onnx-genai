# Wallace — History (compacted 2026-07-29)

**Role:** CUDA EP GEMV/kernel author and reviewer for the Rust runtime — sub-4-bit and IQ decode kernels, GQA/RMSNorm/SiLU parity, and native-serving safety. Verify bit-exactness against CPU, keep kernels correct and fast across all supported SM architectures, and honor reviewer lockouts.

## Durable lessons
- CUDA architecture strings must derive from the live selected device capability (SM60–SM120), never hardcoded; retain the native-CUBIN fallback for when the driver rejects a PTX ISA (e.g. CUDA 13.3 PTX ISA 9.3 on driver 580.105.08). All CUDA kernel work must remain correct and fast across supported SMs, not only `sm_90`.
- IQ super-block M=1 GEMV is bit-exact vs CPU for IQ4_XS/IQ2_XXS/IQ3_XXS/IQ2_XS/IQ2_S/IQ3_S/IQ1_S/IQ1_M; M>1 and unknown formats must fall back. Shared `IQ1S_GRID` hash is `0x6703ed863501ae2e`.
- Native CUDA serving must fail closed: Roy's CUDA-only `559c46f` was rejected/locked out because a real 144-BQMM model failed mid-serving without CPU fallback; Deckard's `fa30410` (startup failure with heterogeneous-placement guidance) is canonical.
- CUDA op order must match CPU's branch-stable SiLU/RMSNorm; a small reduction-order drift (token-16 `1.9073486e-5`) is accepted only because exact emulation costs ~8.4%. Stale CUDA tests must assert the real unsupported error so they cannot pass if the dtype later becomes accepted.
- Repository-wide serialized custom-operator domain is `pkg.nxrt`. Session EP claim planning preserves omitted optional inputs as `DataType::Undefined` (`848ad87`).
- CI covers all 27 offline crates with warnings-as-errors and native Windows ARM64; wave-2 native fp16 CUDA decode reached 663–672 tok/s on H200 vs ORT GenAI 657 with zero fallbacks.

## Recent work (current wave, ~2026-07-28)
## 2026-07-28T17:40:00+0000
Approved PR #365 after four rounds, including recursive scanning and uniform structural identity.

Full pre-compaction history in `history-archive.md`.

## 2026-08-17T16:25Z — PR #1134 validation and ORT-gap measurement

- Measured native base decode ahead of ORT-CUDA on int4 Phi-4-mini (1.33×), qwen7b (1.13×), and qwen14b (1.80×), with mandatory caveat that native used captured decode while the ORT harness was eager. Profiling showed int4 MatMulNBits GEMV dominates decode (~77% GPU time).
- Independently validated Luv's GEMV pipeline: qwen14b 157.39→166.22 tok/s (+5.6%), qwen7b 306.04→312.80 tok/s (+2.2%), and pipe ON/OFF greedy streams were identical over 300 tokens per model.
- Outcome: validation green; PR #1134 merged.

## 2026-08-17T17:10Z — Independent validation: gateup-vec (PR #1137), GREEN

- Validated `e54cae31` in a fresh detached worktree (did not reuse Luv's GPU3); pinned idle GPU5/6; 5 process invocations per config, measurement only.
- Byte-identity (decisive gate): 0.000% divergence over 300 greedy tokens on both qwen2.5-14b and qwen2.5-7b → GREEN.
- Perf: qwen7b +1.4% clean (313.55 vs 309.24, no overlap); qwen14b break-even within noise (164.75 vs 165.37, median −0.4%). Luv's +0.82%/14b did NOT reproduce — 14b variance (~±3 tok/s) exceeds a sub-1% effect; big-model decode is memory-latency-bound, so −7.4% ALU ops has limited E2E leverage. Recommended not advertising a headline 14b number. Merged as `70cc06ad`.

## 2026-08-17T18:05Z — Independent validation: gateup-occ (PR #1139), GREEN

- Validated `squad/gateup-rms-stage @ 11a01fae` in a fresh detached worktree `.worktrees/wallace-gateup-occ` (did NOT reuse Luv's worktree/GPU); pinned idle H200 GPU5 (14b) / GPU6 (7b); measurement only. Interleaved A/B, 5 rounds (OCC=1 then OCC=0 back-to-back, `--steady --tokens 128 --warmups 1 --runs 3`) to control host/thermal drift.
- Byte-identity (decisive gate): 0.000% divergence over 300 greedy tokens on both qwen2.5-14b and qwen2.5-7b → GREEN.
- Perf: 14b OCC=1/OCC=0 median 171.19/166.82 = 1.026x (+2.6%), every round OCC=1 wins (HELD 5/5, min-OCC1 168.46 > max-OCC0 168.29) — confirms Luv's +2.4%. 7b 313.76/313.88 = 0.9996 (~flat, ±0.2%, no regression — 7b GEMV not occupancy-bound at that size).
- Verdict GREEN / MERGE-ELIGIBLE — byte-identical both models, +2.6% on 14b, no 7b regression. Did NOT run ncu (optional; regs 32/82% is Luv's isolated claim) — E2E + byte-identity both pass decisively. Merged as `0636a759`.

## 2026-08-17T18:40Z — Independent validation: gateup-preperm safety GREEN, perf NO-GO

- Validated `squad/gateup-preperm @ 6629f0aa`: byte-identity 0.000% over 300 greedy tokens on both qwen2.5-14b and qwen2.5-7b. Perf did not reproduce: 14b +0.27% median with 2/5 regressions and widened variance; 7b flat. Recommended no default-ON; coordinator shelved/not merged.

- 2026-08-17T21:40Z — Assigned ORT CUDA-graph fairness follow-up to verify native-vs-ORT decode claims under equal graph-capture conditions.
## 2026-08-17T22:20Z — ORT CUDA-graph fairness verdict

- Completed the native-vs-ORT CUDA-graph fairness study. Fixed ORT auto-discovery by pinning `ONNX_GENAI_ORT_LIB` to CUDA ORT and using `ONNX_GENAI_EP_FALLBACK=1` for shape nodes.
- Found stock ORT CUDA EP cannot capture a replayable graph on these int4 genai decode models; graph-vs-graph is therefore an honest capability gap.
- Eager-vs-eager medians: native 1.28× ORT on qwen14b only via on-GPU argmax, but 0.76× on qwen7b and 0.86× on Phi; kernel-only native is 0.75–0.87× ORT. Verdict: native's production lead is graph-capture + device-sampling architecture, not per-kernel speed.
## 2026-08-18T01:35Z — V2-Lite MoE measurement + graph-capture scope

- Measured corrected DeepSeek-V2-Lite int4 MoE: native CUDA eager median ~55.6 tok/s; graph flag currently no-ops because capture declines on `attention_mask_consumers_are_capacity_aware`; ORT CUDA lacks QMoE kernels and falls back to CPU experts at ~0.20 tok/s.
- Scoped V2-Lite graph-capture unlock as GO: topology-gated capacity policy for additive-mask-builder → capacity-form `Attention[3]`, with GLM-5.2 negative guard intact; post-implementation Wallace owns byte-identity/perf A/B.

## 2026-08-18T03:15Z — V2-Lite graph-capture classifier revised and final A/B green

- Under reviewer lockout, revised Deckard's rejected V2-Lite additive-mask capacity classifier by requiring present KV inputs and rejecting root graph-output mask escapes; Rachael re-approved and #1171 merged as `bc1e97ff`.
- After Leon's `_d1` planner fix, measured the real 27-layer DeepSeek-V2-Lite int4 QMoE artifact: capture ON vs eager OFF was byte-identical over 320 tokens and median 101.80 vs 56.94 tok/s (1.79×), with every capture run faster than every eager run.
- Flagged a separate long-context Engine Attention workspace under-plan for Leon; it reproduces with graph capture disabled and is not a classifier/capture regression.
## 2026-08-18T04:15Z — DeepSeek-V2-Lite Native-vs-ORT row closed

- Measured the real 27-layer DeepSeek-V2-Lite int4 QMoE artifact with pinned ORT CUDA 1.27 and confirmed native CUDA serves it on GPU at 57.15 tok/s eager / 101.68 tok/s captured.
- ORT CUDA EP cannot place the 26 `com.microsoft::QMoE` nodes on GPU; with fallback it bridges 104 CPU/GPU Memcpy nodes and reaches 0.17 tok/s, while strict no-fallback refuses the graph.
- Durable framing: this is a hard GPU capability gap (native GPU vs ORT CPU fallback), not a per-kernel speedup claim; ORT CUDA graph is categorically N/A on the split CPU/GPU graph.
