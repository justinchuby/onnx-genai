# Decision: Prompt-lookup speculative decoding on a large decode-bound single-model decoder — KILL

**Author:** Deckard (Systems Dev)
**Date:** 2026-08-14
**Slug:** deckard-spec-14b-verdict
**Related:** PR #932 (spec bench flag), PR #935 (M=K verify near-tie root-cause), Blocker-2 SWA classifier PR #848

## Question
Does native prompt-lookup (n-gram) speculative decoding give a real net win (>1.2×) on a
**large, decode-bound, single-model decoder** — as opposed to the overhead-bound qwen0.5b
(600 tok/s greedy) measured earlier? This is the call-deciding number: YES → wire speculative
into the PipelineEngine path and make it the headline single-stream lever. NO → shelve
speculative; single-stream decode wins must come from Marlin relayout / capture-preserving
architecture changes.

## Testbed substitution
Requested testbed qwen2.5-14b-int4 at `.worktrees/holden-zerocopy/qwen14b-zp` was **incomplete**
(688 KB `model.onnx` graph stub only — no `model.onnx.data` weights, no tokenizer, no
inference_metadata, no active download). Holden (owner) messaged; no usable path returned.
Substituted **glm-4-9b-int4-cuda** (`/home/justinchu/glm-e2e-artifacts/glm-4-9b-int4-cuda`,
READ-ONLY): a complete dense 9B int4 single-model decoder (6.3 GB weights + inference_metadata.yaml
+ tokenizer.json, no pipeline subdirs). Genuinely decode-bound: greedy 10.4 ms/token, ~6.8× the
0.5b's 1.66 ms/token.

## STEP 0 (gate) — PASSED
glm-4-9b loads via the single-model `Engine::from_dir` path (tokenizer.json mandatory,
genai_config optional, inference_metadata accepted), i.e. the speculative-capable
decode_verify/rewind path — NOT PipelineEngine. Speculative is reachable via `--steady`.

## Method
Bench: `profile_native` (`--steady`, `--speculative prompt-lookup --spec-ngram N --spec-tokens K`,
flag from PR #932). tokens=256, warmups=1, runs=4, decode-skip=8. GPU **7** (verified idle;
`nvidia-smi` re-checked — all 8 GPUs idle, GPU 0 had 551 MiB from another user, avoided).
`source .cudaenv.sh`. Build: `cargo build --release -p onnx-genai-bench --features bench-native,cuda --bin profile_native`.

## Results (glm-4-9b, GPU 7)

| Workload | Config | tok/s (median) | vs greedy | Acceptance | tok/verify | invalidations |
|---|---|---|---|---|---|---|
| — | **greedy baseline** | **96.4** (stable) | 1.00× | — | — | 6 |
| Verbatim-copy (max accept) | spec ngram3 K4 | 45.25 | **0.47×** | **96.1%** | 4.84 | 280 |
| Prose (hash-map) | spec ngram3 K4 | 49.0 (9.8–49, unstable) | **0.51×** | 72.9% | 3.92 | 250 |
| Repetitive (5-item list) | spec ngram3 K4 | 71.7 (stable) | **0.74×** | 45.0% | 2.80 | 90 |
| Repetitive (5-item list) | spec ngram2 K5 | 41.6 | 0.43× | 16.0% | 1.80 | 295 |

Greedy: captures=6 replays=1267 invalidations=6 (capture is the entire perf mechanism).
Every spec config: replays collapse (25 in the copy case), invalidations explode (90–295).
Spec output stayed coherent; verbatim-copy case was byte-identical to greedy.

## VERDICT — KILL. No net win on any workload/config; best case 0.74×, typically ~0.5×.

**Acceptance is NOT the bottleneck.** The decisive datapoint: at **96.1% acceptance**
(near-theoretical-max, 4.84 tokens accepted per verify), speculation is still **0.47× greedy —
2× SLOWER**. Raising acceptance does not rescue it.

**Root mechanism.** Native decode here is **DISPATCH-bound** (~1600 kernel launches/token, GPU
~99% idle), not compute-bound. Greedy achieves 96 tok/s *because of CUDA-graph capture*
(replays=1267). The eager M=K `decode_verify` (native_decode/cuda.rs:1229-1312) (a) abandons /
invalidates capture for the verify step (invalidations 6→280, replays 1267→25) and (b) issues ~K×
uncaptured kernel launches. So an eager verify step costs several× a captured M=1 step, and it also
forces re-capture churn on the surrounding M=1 steps. Committing multiple tokens/verify cannot
amortize a cost that is dispatch, not arithmetic — speculation directly fights the exact mechanism
(graph capture) that makes native decode fast.

**The size hypothesis is directionally right but asymptotes below 1.0.** The vs-greedy ratio
improves with model size (0.5b ≈ 0.18× → 9b best 0.74×), confirming per-token cost amortizes the
eager verify better on bigger models — but it plateaus **under 1.0×**, never near the 1.2× win bar,
because the verify step is uncaptured on a dispatch-bound stack. Extrapolating, a 14b/32b would
land ~0.8–0.9× at best; it would not cross 1.2× without a capture-stable verify.

**Instability.** Spec throughput is pathologically variable on prose (49 → 9.8 tok/s run-to-run)
from capture-invalidation storms; greedy is rock-stable.

## Recommendation
1. **Do NOT wire speculative into the PipelineEngine path.** It is not the single-stream lever.
2. Pursue **Marlin relayout / capture-preserving** decode changes for single-stream tok/s.
3. Speculation could only win here if the **verify step were itself capture-stable** (a padded,
   fixed-shape, captured M=K verify graph) so it stops abandoning capture — a substantial build,
   and even then amortization is marginal on dispatch-bound decode. Not worth it now.
4. This also gates EAGLE-3/MTP: any method reusing the eager M=K verify inherits the same
   capture-abandonment penalty. Same prerequisite (capture-stable verify) applies.
