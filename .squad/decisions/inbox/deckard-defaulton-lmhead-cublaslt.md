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
- **GEMV fp16 (#996) NOT flipped** — not present in this base (still in Chew's
  accuracy gate). One-liner hook when it lands: invert its `use_gemv_fp16()`
  gate the same way this PR inverts `lmhead_cublaslt_enabled()`. Bonus rationale
  for #996 default-on: fp16 GEMV is deterministic, which also fixes the fp32
  default path's run-to-run nondeterminism Gaff flagged.

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

## Note on the 232 target

The ~232 combined number (deckard-combined-stack-measurement) required #996's
fp16 GEMV, which is NOT in this base. Off plain main, default-on banks #991 +
#999 only => **~208 short / ~172 KV2048**. The remaining jump to 232/182 is the
one-liner #996 fp16-GEMV default flip after its accuracy gate passes.

## Gates
- f64 oracle 7/7; GQA fp16 2/2; `cargo fmt` + `cargo clippy` clean.
- Did not touch `platform_capacity.rs:247/249` (pre-existing clippy, per coord).
