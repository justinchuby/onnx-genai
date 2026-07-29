# Resch: MLAS QNBit wire-up study

Date: 2026-07-29
Branch: qwen3-perf-followups
Scope: onnx-runtime-ep-cpu only; mlas-sys left untouched after Deckard's 84bdb3ba.

## What changed

- Added an explicit `ONNX_GENAI_CPU_MM_MLAS_QNBIT` override:
  - unset: keep the established fastest defaults;
  - `0`/`off`: disable all MLAS QNBit routing;
  - `1`/`on`: opt non-Apple ARM64 decode into MLAS/KleidiAI QNBit for A/B.
- ARM64 CompInt8 M=1 asymmetric zero-point is no longer blocked by the old x86 AVX512-only guard; aarch64 is allowed after Deckard's fix.
- Added a non-Apple ARM64 reachability test proving bits=4 and bits=8, block128, asymmetric qzeros, accuracy_level=4, M=1 route through MLAS QNBit when opted in.
- Fixed the MLAS `NO_SHARD` comparison path so full-width packed B is cached once instead of repacked every decode call.

## Selection result

The MLAS QNBit path is correct/reachable, but it was not the fastest default in this window. Sharded native+MLAS did not match ORT; full-width MLAS was much slower under the native decode pool. Therefore the default stays the already-shipped native KAI packed-SDOT path for non-Apple aarch64+dotprod, with MLAS QNBit available by opt-in for further investigation.

Likely reason: ORT drives MLAS/KleidiAI inside its own execution/threading model. Our per-worker N-sharded calls add native decode-pool overhead and lose enough efficiency that the same ukernel does not automatically match ORT. A single full-width MLAS call with MLAS internal threading oversubscribed/underperformed badly in this process even after caching packed B.

## Benchmarks

Command shape: `profile_native --backend {native,ort} --steady --warmups 1 --runs 5 --tokens 128`.
This machine was noisy during the run (large prefill/decode outliers), so medians below are an honest measurement of this window, not a clean repeat of the earlier 96.3 tok/s KAI result.

| Model | Config | tok/s median | Roofline | % roofline | % ORT |
|---|---:|---:|---:|---:|---:|
| qwen3-0.6b | native+MLAS opt-in sharded | 62.60 | 211 | 29.7% | 61.3% |
| qwen3-0.6b | native default KAI | 79.28 | 211 | 37.6% | 77.6% |
| qwen3-0.6b | native fallback | 32.59 | 211 | 15.4% | 31.9% |
| qwen3-0.6b | ORT | 102.15 | 211 | 48.4% | 100% |
| qwen2.5-0.5b | native+MLAS opt-in (block32 remains KAI/default) | 96.80 | 344 | 28.1% | 53.6% |
| qwen2.5-0.5b | native default KAI | 68.10 | 344 | 19.8% | 37.7% |
| qwen2.5-0.5b | native fallback | 86.58 | 344 | 25.2% | 48.0% |
| qwen2.5-0.5b | ORT | 180.47 | 344 | 52.5% | 100% |
| qwen3-1.7b | native+MLAS opt-in sharded | 18.69 | 76.5 | 24.4% | 58.7% |
| qwen3-1.7b | native default KAI | 12.76 | 76.5 | 16.7% | 40.1% |
| qwen3-1.7b | native fallback | 13.84 | 76.5 | 18.1% | 43.5% |
| qwen3-1.7b | ORT | 31.83 | 76.5 | 41.6% | 100% |

Earlier clean KAI milestone remains the best known native default: 57 -> 71 -> 83 -> 96.3 tok/s on qwen3-0.6b (45.6% roofline, 91% of ORT 105.7), from packed-no-expansion, N16 retile, and default-on KAI.

## Default decision

Do not enable MLAS QNBit by default yet. Keep KAI default-on for non-Apple aarch64+dotprod because it is the best validated native-EP speedup and avoids the MLAS threading mismatch. Keep Apple/x86 dispatch unchanged. Use `ONNX_GENAI_CPU_MM_MLAS_QNBIT=1` to continue MLAS/KleidiAI experiments.

## Remaining lever

To make MLAS beat or match ORT we need to remove the native/MLAS threading mismatch: either call the exact ORT-style QNBit batch shape/threading, or add a lower-overhead native batching path that lets one MLAS invocation cover all N tiles without per-token repack or decode-pool oversubscription. If that still trails ORT, fall back to Luba's hand-scheduled NEON asm plan for the KAI N16/N32 kernel.
