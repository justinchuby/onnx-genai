# Decision: Native prompt-lookup speculation is not a win as shipped (measured)

**Author:** Deckard (Systems Dev)
**Date:** 2026-08-14
**Branch:** squad/bench-prompt-lookup
**Status:** proposed

## Context
Task: benchmark the shipped-but-unmeasured native prompt-lookup (n-gram) speculative
decode path (WP2). Added a minimal opt-in bench flag to `profile_native`
(`--speculative prompt-lookup --spec-ngram N --spec-tokens K`) — no engine changes.

## Measurement setup
- GPU: **CUDA_VISIBLE_DEVICES=7** (verified idle, H200; re-checked `nvidia-smi` before runs).
- Model: `/home/justinchu/qwen2.5-0.5b-int4-onnx` (single-model int4 qwen2, native CUDA EP).
  - **Muse-Glimmer-30B could NOT be used** — see finding 1.
- Harness: `profile_native --ep cuda --backend native --steady --tokens 200 --warmups 1 --runs 2 --decode-skip 8`.
- Sweep: ngram∈{2,3} × K∈{3,4,5}, two workloads (repetitive prose, general prose).

## Findings

### 1. STRUCTURAL: prompt-lookup is unreachable on the PipelineEngine path (headline)
Native speculation (`NativeSpeculativeDriver`, `decode_verify`+`rewind`) is wired ONLY
into the single-model `Engine` path (`engine/runtime.rs`). The `PipelineEngine`
autoregressive path (`pipeline/flat_autoregressive.rs`) drives decode through the strict
one-token-per-step `run_decode_loop` / `DecodeLoopBackend` trait (`decode_loop.rs:56`),
which has **no k-token verify/rewind hook**. Muse-Glimmer-30B is a multi-component
pipeline (embedding + decoder + vision_encoder) → requires `--pipeline` → **cannot use
prompt-lookup as shipped**. Every VLM/pipeline model is in the same boat. Wiring
speculation into the pipeline decode loop is a separate engine task (out of scope here).

### 2. CORRECTNESS: prompt-lookup is NOT byte-lossless vs greedy (bug)
Every (ngram,K) config diverged from the greedy token stream at a fixed index
(prose idx 27, repetitive idx 58 — the first verify step), deterministically and
reproducibly. Root cause isolated: with CUDA graph **disabled**, greedy is byte-identical
to graph-on greedy, but speculative still diverges identically → the divergence is **not**
capture-related; it comes from the **eager M=K batched verify** producing a different
argmax than the sequential M=1 decode. "Exact verification" is only lossless if the M=K
verify logits equal the M=1 logits; on this GPU/model they don't. This violates the
losslessness guarantee and should be treated as a bug in the verify path.

### 3. PERFORMANCE: large net loss on this model, best-case included
Greedy baseline: **591–603 tok/s** (decode 1.66 ms/token, cuda_graph invalidations=3–4).

Repetitive workload (greedy 591.57 tok/s):
| ngram,K | tok/s | vs greedy | accept% | tok/verify | graph invalidations |
|---------|-------|-----------|---------|------------|---------------------|
| 2,3 | 72.77 | 0.12x | 64.0 | 2.92 | 216 |
| 2,4 | 70.61 | 0.12x | 41.1 | 2.63 | 261 |
| 2,5 | 73.50 | 0.12x | 33.8 | 2.69 | 264 |
| 3,3 | 84.66 | 0.14x | 63.1 | 2.88 | 186 |
| 3,4 | 109.17 | 0.18x | 71.5 | 3.84 | 135 |
| 3,5 | 104.00 | 0.18x | 53.8 | 3.69 | 156 |

General prose workload (greedy 602.97 tok/s):
| ngram,K | tok/s | vs greedy | accept% | tok/verify | graph invalidations |
|---------|-------|-----------|---------|------------|---------------------|
| 2,3 | 93.48 | 0.16x | 25.0 | 1.75 | 209 |
| 2,4 | 83.88 | 0.14x | 29.4 | 2.17 | 221 |
| 2,5 | 82.66 | 0.14x | 22.7 | 2.12 | 236 |
| 3,3 | 139.09 | 0.23x | 45.7 | 2.33 | 119 |
| 3,4 | 160.28 | 0.27x | 48.7 | 2.95 | 101 |
| 3,5 | 136.93 | 0.23x | 39.8 | 2.95 | 104 |

Best config both workloads = ngram=3,K=4. Even at 71.5% acceptance, speculative is
**3.8–5.4x SLOWER** than greedy.

### 4. CAPTURE INTERACTION: verify thrashes the graph
Greedy: captures≈3–4, invalidations≈3–4. Speculative: invalidations 100–264, decode
6–18 ms/token (vs 1.66 greedy). The eager M=K verify step invalidates/re-captures the
CUDA graph each time it fires; on this small model the captured M=1 path is so cheap
(1.66 ms) that the eager verify overhead dwarfs any savings from multi-token accepts.
Prefill also jumps to ~45–200 ms (from ~4 ms).

## Verdict
Prompt-lookup speculation is **NOT a free win** on this model — it is a large net loss on
BOTH repetitive (best case) and prose (typical) workloads, AND it is not byte-lossless.
Do not enable it by default.

**Caveat:** qwen2.5-0.5b is tiny and dispatch/overhead-bound, so the eager verify penalty
is worst-case here. On a large decode-bound model (30B) each saved forward pass is worth
much more and the verify arithmetic amortizes better, so the perf tradeoff *could* differ.
I could not verify that: Muse-Glimmer (pipeline) can't run the feature (finding 1), and no
large single-model native decoder loaded via `Engine::from_dir` (granite-1b / gemma4-e2b
lack genai_config; onnx-27b load exceeded a 280 s budget). The two model-independent
blockers stand regardless: (a) unreachable on the pipeline path; (b) not byte-lossless.

## Recommendation
1. Before any perf work, FIX the M=K-verify non-losslessness (finding 2), or explicitly
   re-scope prompt-lookup as approximate.
2. Do NOT enable prompt-lookup for pipeline/VLM models — it silently no-ops (guarded in
   the bench).
3. Re-measure on a large single-model decode-bound model once one is loadable to decide if
   it's ever a win; keep it opt-in and off by default until then.
