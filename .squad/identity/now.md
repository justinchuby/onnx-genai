# Team Focus — now

**Current focus:** glm-4-9b-int4 CUDA decode-vs-ORT program on H200. Native base decode is now **192.4 tok/s** after #986, with this arc moving **97.5→112→138→192 tok/s** and narrowing the ORT base gap from **2.57× to ~1.30×**. Honest standing: native base still **loses** to ORT base (~250 tok/s). Speculative decode is additive-on-top and is **not** the basis for an ORT-win claim.

## Merged this arc

- **#978 (`532ef6bc`) — int4 M=1 decode GEMV split-K:** LOP3 dequant + grid-fill split-K for the generic block-size path. glm **97.5→112.4 tok/s** (+15%); qwen block-32 unchanged. A/B: `ONNX_GENAI_GENERAL_SPLITK`.
- **#981 (`b24e961e`) — decode SkipRMSNorm block kernel:** multi-warp block parallelization for M=1 / M≤8 verify. glm **112.3→137.8** (+22.7%); qwen **125.6→148.85** (+18%). A/B: `ONNX_GENAI_CUDA_DISABLE_SKIP_RMSNORM_BLOCK`.
- **#986 (`e8f76c53`) — 128-bit wide-load int4 GEMV:** default-on `uint4` wide-load path for glm-class block-128 general-BS GEMV. glm **140.7→192.4 tok/s** (+36.7%); Gaff reproduced +35%; Chew f64 oracle **7/7** and glm/qwen byte-identical. A/B: `ONNX_GENAI_GEMV_WIDELOAD=0`.

All landed default-on, portable, capture-clean, and byte-identical at greedy-token level.

## In flight

- **Deckard — `squad/int4-gemv-wideload-v2`:** base GEMV-v2 to deepen memory-level parallelism and push native int4 GEMV streaming from **1.40→2.42 TB/s** (ORT level), aiming to beat ORT base decode (~250 tok/s; target around ~280 tok/s if full streaming is captured).
- **Batty — `squad/spec-decode-w9-contract`:** fix the W=9 speculative-decode contract after #988. This is a secondary on-top layer: captured spec must match plain M=1 greedy before it can stack on base decode.

## Killed / learned

- **cp.async M=1 int4 GEMV is a NO-GO** (#980 closed, not merged): every config regressed −15% to −20%. M=1 has ~8 FMA per loaded word, so async-copy machinery has no compute to overlap.
- The earlier “base-decode floor” verdict is **overturned**. ORT streams the same int4 GEMV at **2.42 TB/s** vs native now **1.40 TB/s** after #986; the remaining base gap is MLP/streaming-limited, not compute-limited or irreducible dequant math.
- **#984 is closed/superseded.** Captured verify had qwen workspace/capture failures and is no longer the active artifact.
- **#988 fixed capture crashes but is rejected.** Gaff 🟢 approved graph-slot/capture stability, but Chew 🔴 rejected qwen W=9 (`spec_tokens=8`) because captured spec diverges from plain greedy at token[2] (**9370 vs 2810**). Deckard is locked out; Batty owns the contract fix.

## Prior waves (compressed)

Earlier August work delivered native CUDA capture, bf16/int4 decode improvements, Marlin M>1 speculative-verify capture, DeepSeek/QMoE fixtures, and Apple/fixture/testing directives. Detailed history lives in `.squad/decisions-archive/2026-08.md` and agent history archives. Old 27B QMoE / LinearAttention status is no longer the current focus.

## Standing facts

- Native-vs-ORT comparisons must use identical weights, optimized CUDA decode configs, steady-state methodology, and oracle/token fairness where applicable.
- Base native decode must be compared against ORT base; speculative decode is additive and cannot be used as the ORT-win basis.
- CUDA-graph decode profiling should use `nsys --cuda-graph-trace=node`; eager op profiling can mis-rank launch-heavy ops.
- Keep shipped decode optimizations byte-identical or oracle-gated, portable/default-on only when Rule 11 is satisfied, and guarded by honest A/B env vars.
