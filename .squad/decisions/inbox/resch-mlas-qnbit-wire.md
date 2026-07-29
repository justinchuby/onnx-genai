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

## 2026-07-29 follow-up: ORT-style full-width driving attempt

Prompt asked whether the problem was our sharded driving rather than the MLAS/KleidiAI kernel. I checked the wrapper:

- `mlas-sys/src/lib.rs::sqnbit_gemm(..., multithread)` calls `ensure_threading()` then `sqnbit_gemm_into`, which passes the boolean through to the shim.
- `vendor/shim.cpp::mlas_qnbit_gemm` maps `multithread != 0` to a non-null `MLAS_THREADPOOL*` sentinel and calls `MlasQNBitGemmBatch<float>(..., thread_pool, ...)`.
- The standalone MLAS hooks (`mlas_set_threading`, `MlasStandaloneMaxThreads`, `MlasStandaloneTrySimpleParallel`) route MLAS's internal partitioner through Rayon. There is no first-class MLAS threadpool object or per-call thread-count argument exposed; the effective thread count is `rayon::current_num_threads()` in the Rayon pool that calls MLAS.

I changed the opt-in ARM64 MLAS path to drive one full-width packed-B call per MatMulNBits node, installed on a bounded Rayon pool sized by `ONNX_GENAI_CPU_DECODE_THREADS` (6/8), instead of N-sharding across native decode workers. Packed B is cached once in `self.mlas_packed`; no per-token B repack on the EP side. I kept this opt-in only (`ONNX_GENAI_CPU_MM_MLAS_QNBIT=1`) because it is diagnostic and slower.

### Clean-window status

The machine did not produce a clean window. ORT was stable (~96-103 tok/s), but KAI sanity failed badly despite waiting/quiescing: KAI median was 11.97 tok/s in one 9-run interleaved pass and 21.58 tok/s after another 120s wait, versus the prior clean 96.3 tok/s. That invalidates absolute native medians from this window.

Still, the relative diagnostic is clear enough: full-width MLAS-internal threading did not approach ORT here.

| qwen3-0.6b config | runs | median tok/s | spread/min-max tok/s | note |
|---|---:|---:|---:|---|
| native+MLAS full-width, `ONNX_GENAI_CPU_DECODE_THREADS=6` | 9 | 34.73 | 31.20-46.80 | one full-width `sqnbit_gemm`, MLAS multithread via 6-thread Rayon pool |
| native+MLAS full-width, `ONNX_GENAI_CPU_DECODE_THREADS=8` | 9 | 28.79 | 7.23-30.36 | 8 threads worse; first run severe outlier |
| native+KAI default | 9 | 11.97 | 3.69-90.21 | sanity anchor failed; window invalid |
| native fallback | 9 | 51.45 | 7.57-75.59 | noisy |
| ORT final | 9 | 102.77 | 100.54-104.92 | stable reference |

### Conclusion / need from Deckard

The Rust wrapper can enable MLAS parallelism, but it cannot expose an ORT-owned `MLAS_THREADPOOL*` or a per-call thread-count-limited MLAS threadpool. It also allocates QNBit workspace inside every `sqnbit_gemm_into` call. If we want true ORT-style driving, Deckard should extend `mlas-sys` rather than Resch hand-editing it:

1. expose a thread-count-limited MLAS/QNBit execution context (or equivalent shim-owned threadpool) so `MlasQNBitGemmBatch` uses exactly 6/8 workers without relying on ambient Rayon state;
2. expose reusable QNBit workspace or a context API so decode does not allocate workspace per MatMulNBits call;
3. optionally expose a batch API matching ORT's call shape exactly if ORT passes more than one batch descriptor or uses different threadpool semantics.

Until then, keep KAI as default and keep MLAS QNBit as opt-in diagnostic only.
