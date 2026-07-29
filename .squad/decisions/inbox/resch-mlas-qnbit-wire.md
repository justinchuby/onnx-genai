# Resch: MLAS QNBit wire-up final state

Date: 2026-07-29
Branch: qwen3-perf-followups
Scope: `onnx-runtime-ep-cpu` only for Resch changes; `mlas-sys` remains Deckard-owned.

## Final summary

We should stop benchmarking on the current host. The machine is thermally/load-saturated: the KAI sanity anchor that previously measured cleanly at ~96 tok/s reproduced at only ~12-22 tok/s in the latest window. That 4-8x native slowdown makes all current native-vs-ORT absolute numbers invalid. The honest final state is: KAI is banked and default; MLAS QNBit is wired and reachable but performance-unvalidated; the decisive MLAS-vs-ORT test needs a quiet machine plus a better `mlas-sys` execution context.

## Banked / valid from clean windows

- Native EP KAI packed-SDOT decode is default-on for non-Apple aarch64+dotprod as of `063216fb`.
- Clean qwen3-0.6b result: native fallback ~69 tok/s -> KAI ~96.3 tok/s.
  - ~1.39x faster than fallback.
  - 45.6% of the 211 tok/s roofline.
  - ~91% of ORT's ~105.7 tok/s reference.
- The clean KAI trajectory was 57 -> 71 -> 83 -> 96 tok/s:
  1. packed-no-expansion qsi4/qsi8 unlocked the first jump;
  2. decode worker saturation and 8-worker default helped;
  3. N16 KAI tile-major retile fixed the SDOT latency bottleneck;
  4. default-on non-Apple aarch64+dotprod banked the native-EP win.
- Deckard banked the MLAS dependency in `84bdb3ba`:
  - full MLAS + KleidiAI vendored;
  - `mlas` feature default-on;
  - ARM64 CompInt8 QNBit access violation fixed;
  - bits4/bits8 block128 asymmetric QNBit round-trip passing on ARM64.

## Wired but not performance-validated

- `6cb8c21` wires a non-Apple ARM64 MLAS QNBit opt-in route for MatMulNBits decode:
  - `ONNX_GENAI_CPU_MM_MLAS_QNBIT=1` opts bits=4/bits=8, block128, asymmetric qzeros, `accuracy_level=4`, M=1 into MLAS QNBit CompInt8;
  - unset keeps the established KAI default;
  - `0`/`off` disables MLAS QNBit routing for A/B;
  - ARM64 is allowed through the old asymmetric CompInt8 M=1 guard after Deckard's fix;
  - reachability tests prove bits4 and bits8 block128 M=1 can select MLAS QNBit when opted in.
- `eba48f58` changes the opt-in MLAS experiment from N-sharded, single-threaded per-shard calls to one cached full-width packed-B QNBit call per MatMulNBits node, installed on a bounded Rayon pool sized by `ONNX_GENAI_CPU_DECODE_THREADS` (6/8). This is closer to ORT-style driving and confirms packed B is cached once in `self.mlas_packed` rather than repacked per token.
- This route is still **unvalidated for performance** because the current host cannot reproduce the KAI sanity anchor. KAI remains the default. MLAS remains opt-in via `ONNX_GENAI_CPU_MM_MLAS_QNBIT=1`.

## Why the latest benchmark window is invalid

The latest attempted clean-window qwen3-0.6b run used 9-run interleaving and explicit 6/8-thread MLAS settings, but the sanity anchor failed:

| Config | Median tok/s | Spread | Status |
|---|---:|---:|---|
| KAI default, expected clean anchor | ~96.3 | prior clean window | valid banked baseline |
| KAI default, latest pass | 11.97 | 3.69-90.21 | invalid/noise-corrupted |
| KAI default, retry after 120s wait | 21.58 | 5.81-80.35 | invalid/noise-corrupted |
| ORT final in same noisy period | 102.77 | 100.54-104.92 | stable reference, but native host state invalid |

Because KAI cannot reproduce its clean 96 tok/s anchor, none of the latest native MLAS/KAI/fallback medians should be used to make a default decision.

## MLAS wrapper/threading finding

The current wrapper can enable MLAS internal parallelism, but it does not expose a first-class ORT-like execution context:

- `mlas-sys/src/lib.rs::sqnbit_gemm(..., multithread)` calls `ensure_threading()` and then `sqnbit_gemm_into`.
- `vendor/shim.cpp::mlas_qnbit_gemm` maps `multithread != 0` to a non-null `MLAS_THREADPOOL*` sentinel and calls `MlasQNBitGemmBatch<float>(..., thread_pool, ...)`.
- The standalone hooks route MLAS parallel-for through Rayon; the effective thread count is `rayon::current_num_threads()` in the pool that calls MLAS.
- There is no exposed per-call MLAS thread-count argument, no real `MLAS_THREADPOOL*` context object, and no reusable QNBit workspace context. `sqnbit_gemm_into` allocates workspace per call.

## Remaining lever for a quiet machine

The decisive next test should happen only on a quiet host where KAI reproduces ~96 tok/s first. Then run the 4-way table in one clean window:

1. native+MLAS QNBit, ORT-style one-call-per-node driving;
2. native+KAI default (sanity anchor must be ~96 tok/s on qwen3-0.6b);
3. native fallback;
4. ORT backend.

Before that test, Deckard should extend `mlas-sys` with the missing execution support:

1. a thread-count-limited MLAS/QNBit execution context or shim-owned `MLAS_THREADPOOL` equivalent so one `MlasQNBitGemmBatch` call uses exactly 6-8 workers without relying on ambient Rayon state;
2. reusable QNBit workspace/context so decode does not allocate workspace per MatMulNBits call;
3. optionally an exact ORT-shaped QNBit batch API if ORT uses more specific batch/threadpool semantics than the current wrapper exposes.

If native+MLAS then matches ORT, it can become the maintained default. If it still trails despite true ORT-style driving on a quiet machine, the remaining gap is ORT's graph/runtime integration and we should consolidate around KAI default plus future hand-scheduled NEON/asm work.

## Default decision

Keep KAI packed-SDOT as the default for non-Apple aarch64+dotprod. Keep Apple/x86 dispatch unchanged. Keep MLAS QNBit as an opt-in diagnostic path (`ONNX_GENAI_CPU_MM_MLAS_QNBIT=1`) until a quiet-machine benchmark with KAI≈96 and a proper `mlas-sys` threadpool/workspace context proves it should replace KAI.
