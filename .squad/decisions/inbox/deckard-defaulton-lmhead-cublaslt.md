# Decision: default-on lm_head cuBLASLt + multi-shape capture guard

**Author:** Deckard (Systems Dev, CUDA/decode-perf)
**Date:** 2026-08-15
**Branch:** `squad/base-wins-default-on` (off main `b550b17c`, which has #991 + #999)
**PR:** (draft — see branch)
**Status:** measured green, awaiting Chew/Gaff gate + merge

## What & why

Bank the landed base-decode wins as the **out-of-box** glm number (no env
flags). This PR promotes the #991 cuBLASLt `lm_head` decode MatMul to
**default-on** and first clears Gaff's #991 blocker: the single-shape cuBLASLt
plan cache is replaced by a **multi-shape (shape-keyed) plan cache** so plan
creation never happens inside a captured region across prefill/decode/verify
shape changes.

## The capture guard (Gaff's #991 blocker)

`MatMulKernel::launch_dense_capturable` previously held a single
`Option<DenseGemmPlan>`: any shape switch evicted + recreated the plan (a
heuristic algo search). Fine for base decode (one prefill->decode switch), but
a spec-decode verify width (M=K) alternating with decode M=1, or a multi-turn
per-turn prefill, would recreate **every step** — and a recreate *inside* the
captured region is illegal (heuristic query + alloc + cache mutation) => capture
break.

Fix: shape-keyed `Vec<DenseGemmPlan>` keyed by `(dtype, M, K, N)`, MRU-ordered,
bounded by `DENSE_PLAN_CACHE_CAP = 8`.
- **Hit** -> promote to MRU front, launch preselected algo. No heuristic query.
- **Cold miss while capturing** -> `Err` (unchanged reject-before-warm contract,
  now enforced per shape). Every shape used in a captured graph must be warmed
  by a preceding non-capturing pass.
- **Cold miss not capturing** -> create, insert at front, LRU-truncate to cap.
- **Eviction safety:** eviction runs *only* on the non-capturing creation path;
  the hot decode plan (M==1) is always MRU-front and never the LRU victim while
  hot. Strictly safer than the single-slot path (evicts far less; same
  recapture-on-shape-change discipline protects the freed-workspace hazard).

## Default-on flip

- `lmhead_cublaslt_enabled()` inverted to **default-TRUE-unless-disabled**.
  Escape hatch: `ONNX_GENAI_LMHEAD_CUBLASLT=0|false|off` -> hand fp16 GEMV.
- **#999 GQA flash-decoding is already default-on** (audited): its env vars
  (`ONNX_GENAI_CUDA_GQA_DIRECT_SINGLE_SPLIT`, `ONNX_GENAI_CUDA_GQA_SPLITS`) are
  rollback/A-B knobs, defaults = the optimized path. No flip needed.
- **GEMV fp16 (#996) stays PERMANENTLY opt-in — NOT flipped, no default hook.**
  Chew's #996 verdict (🟡 land-as-opt-in): the fp16 K-accumulate is not
  byte-identity-safe vs the fp32 path — glm has intermittent knife-edge argmax
  flips (English↔Chinese at idx 3, dup at idx 12; ~2 in first 4 runs then 0
  across 70+). So fp16 GEMV is not a default-on candidate and the earlier
  "one-liner hook to flip it later / fp16 fixes fp32 nondeterminism → default it"
  plan is **withdrawn**. Out-of-box GEMV stays fp32 (byte-identical). fp16
  remains available via `ONNX_GENAI_GEMV_FP16=1` for users who opt in.

## Measured (glm-4-9b-int4, H200 GPU3, graph-on, median)

| config | short tok/s | KV2048 tok/s | fallbacks |
|---|---|---|---|
| **default-on (no flags)** | **207.72** | **171.69** | **0** |
| explicit `=1` | 207.31 | — | 0 |
| escape-hatch `=0` (hand GEMV) | 202.24 | — | 0 |

- KV2048 default-on: fallbacks=0 across **8623 replays / 8 bucket-growth
  recaptures** — the multi-shape cache holds through prompt->decode->bucket
  growth. This is the capture-guard proof.
- **Byte-identity:** cuBLASLt lm_head (default) vs hand-GEMV (off) top-40
  token-0 logprobs are **byte-identical** (both select token 315).
- **qwen no regression:** 152.16 (on) vs 153.60 (off), within noise, fallbacks=0.

## Note on the 232 number (out-of-box is ~208, and that is final)

The ~232 combined number (deckard-combined-stack-measurement) required #996's
fp16 GEMV **opt-in ON**. Since fp16 GEMV now ships **permanently opt-in** (Chew
knife-edge flip — not byte-identity-safe for default), the honest out-of-box
default-on config is **fp32 GEMV + cuBLASLt lm_head + flash-decoding =
~207.7 short / ~171.7 KV2048** (measured here, fp16 OFF). 232 remains reachable
only by a user setting `ONNX_GENAI_GEMV_FP16=1`; it is not the default.

## Gates
- f64 oracle 7/7; GQA fp16 2/2; `cargo fmt` + `cargo clippy` clean.
- Did not touch `platform_capacity.rs:247/249` (pre-existing clippy, per coord).
