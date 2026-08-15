# Team Focus — now

**Current focus:** glm-4-9b-int4 CUDA decode-vs-ORT program on H200. Native base decode is now **137.8 tok/s** after the +41% arc, with the ORT gap narrowed from **2.57× to 1.82×** against stock ORT (~250 tok/s) on identical weights.

## Merged this arc

- **#978 (`532ef6bc`) — int4 M=1 decode GEMV:** LOP3 dequant + grid-fill split-K for the generic block-size path. glm **97.5→112.4 tok/s** (+15%); qwen block-32 unchanged. A/B: `ONNX_GENAI_GENERAL_SPLITK`.
- **#981 (`b24e961e`) — decode SkipRMSNorm block kernel:** multi-warp block parallelization for M=1 / M≤8 verify. glm **112.3→137.8** (+22.7%); qwen **125.6→148.85** (+18%). A/B: `ONNX_GENAI_CUDA_DISABLE_SKIP_RMSNORM_BLOCK`.

Both landed default-on, portable, capture-safe, and byte-identical at greedy-token level.

## Killed / learned

- **cp.async M=1 int4 GEMV is a NO-GO** (#980 closed, not merged): every config regressed −15% to −20%. M=1 has ~8 FMA per loaded word, so async-copy machinery has no compute to overlap.
- The earlier “base-decode floor” verdict is **overturned**. ORT streams the same int4 GEMV at **2.42 TB/s** vs native **0.92 TB/s** with similar tiling and occupancy; the remaining base gap is narrow 32-bit loads / low memory-level parallelism, not irreducible dequant math.

## In flight

- **Deckard — `squad/int4-gemv-wideload` (GPU6):** 128-bit synchronous wide-load int4 GEMV, preserving per-lane accumulation order. Target: partial **180–200 tok/s**, full ORT-like base parity around **236 tok/s**.
- **Sebastian — `squad/spec-decode-e2e` (GPU7):** speculative-decode e2e using captured M=8 verify and selective KV commit. Captured-vs-eager M=8 logits are byte-identical on glm/qwen; glm B*≈2.16 is a practical GO. This is the multiplicative lever to pass ORT after base GEMV improves.

## Prior waves (compressed)

Earlier August work delivered native CUDA capture, bf16/int4 decode improvements, Marlin M>1 speculative-verify capture, DeepSeek/QMoE fixtures, and Apple/fixture/testing directives. Detailed history lives in `.squad/decisions-archive/2026-08.md` and agent history archives. Old 27B QMoE / LinearAttention status is no longer the current focus.

## Standing facts

- Native-vs-ORT comparisons must use identical weights, optimized CUDA decode configs, steady-state methodology, and oracle/token fairness where applicable.
- CUDA-graph decode profiling should use `nsys --cuda-graph-trace=node`; eager op profiling can mis-rank launch-heavy ops.
- Keep shipped decode optimizations byte-identical or oracle-gated, portable/default-on only when Rule 11 is satisfied, and guarded by honest A/B env vars.
