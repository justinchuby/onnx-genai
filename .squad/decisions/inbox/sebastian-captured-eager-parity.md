# Decision drop — captured-vs-eager M=8 token parity (closing the 2×2)

**Author:** Sebastian (Performance Engineer, CUDA & Perf)
**Branch:** `squad/marlin-bench-captured-parity` (off main `a6030630`, frozen Marlin kernel = code `3735d57e`)
**Context:** Deckard's final ask before Chew's #960 numerics review — fill the last cell of the
captured/eager × glm/qwen 2×2 with an explicit **qwen captured-M=8 vs eager-M=8 token-equality**
data point. Marlin M>1 default (`ONNX_GENAI_MARLIN_M_GT_1=1`, split-K default-on).

## Why a new harness cell was needed
- The shipping speculative decode path uses the **eager** verify path (`native_decode/backend.rs`:
  *"the eager verify path captures nothing"*), so there is no shipping path that emits "captured
  M=8 real tokens." The captured M=8 verify graph is a **probe-only** capability (leverb Part D).
- The existing leverb INC0 probe writes a **constant** token (timing/segments/B\* only) and never
  compared captured vs eager output → it could not answer the token-equality question.
- The existing e2e parity test (`marlin_m_gt_1_e2e`) compares Marlin **eager** vs tiled **eager**
  (the eager cell) — not captured-vs-eager.

## What I added (capture-validation harness, my lane)
- `native_decode/cuda.rs`: `leverb_increment0_token_parity_attempt(m, &tokens)` — a parity variant
  of the INC0 capture attempt that (1) writes **real** prompt tokens (cycled to fill M rows, avoids a
  constant-token near-tie), (2) snapshots the **eager** pre-capture warm-forward logits, (3) captures
  + replays the identical device bindings, (4) compares per-row greedy argmax **and** raw logits
  bytes. Deterministic (same kernel, same inputs) → byte-identical expected. Gated
  `#[cfg(all(test, feature="cuda"))]`, no change to any shipping path.
- `native_decode/leverb_phase0_probe.rs`: **Part E** runs the parity attempt at M=8 and M=1 and
  prints per-row eager vs capture argmax + `logits_byte_identical` + PASS/FAIL.

## Result — full 2×2 GREEN, captured == eager BYTE-IDENTICAL on both models

| Model (block) | M | argmax_match | logits_byte_identical | segments | B\* (M=8/M=1) |
|---|---|---|---|---|---|
| **qwen2.5-14b** (block-32) | 8 | ✅ true | ✅ true | 1 (whole-graph) | 4.45× (35.5/7.98 ms) |
| **qwen2.5-14b** | 1 | ✅ true | ✅ true | 1 | — |
| **glm-4-9b** (block-128, canonical) | 8 | ✅ true | ✅ true | 1 (whole-graph) | 2.19× (22.2/10.2 ms) |
| **glm-4-9b** | 1 | ✅ true | ✅ true | 1 | — |

- qwen M=8 eager argmax `[103572,119785,120095,67545,115971,112845,8138,115971]` == capture argmax.
- glm  M=8 eager argmax `[3832,98559,13861,59265,59265,146200,73053,59265]` == capture argmax.
- **Captured-vs-eager parity is byte-identical** (stronger than token-equality; no tie-robustness
  needed). This closes the 2×2: glm captured✓/eager✓ · qwen captured✓/eager✓ + Marlin-determinism✓.
- B\* refresh is consistent with the frozen plateau: glm ≈2.16–2.19× (practical GO), qwen ≈4.45–4.7×
  (second-model small-M `mma.m16n8k16` floor — non-blocking drafting-depth story, not a GEMM bug).

## Reproduce (verified-idle high-index GPU; contention-invariant)
```bash
source /home/justinchu/onnx-genai/.cudaenv.sh
# re-check: nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader ; pick idle high idx
CUDA_VISIBLE_DEVICES=7 \
ONNX_GENAI_RUN_CUDA_SMOKE=1 ONNX_GENAI_MARLIN_M_GT_1=1 \
ONNX_GENAI_LEVERB_MODEL=/home/justinchu/shared-models/qwen2.5-14b-instruct-int4-zp-onnx \
cargo test -p onnx-genai-engine --features cuda,native-backend \
  --lib leverb_phase0_capture_probe -- --ignored --nocapture 2>&1 | grep '\[leverb-phase0\]\[E\]'
# glm: swap ONNX_GENAI_LEVERB_MODEL=/home/justinchu/glm-e2e-artifacts/glm-4-9b-int4-cuda
```

## Verdict for the merge package
The captured-vs-eager cell is now explicit and **byte-identical** on both glm (canonical) and qwen.
Combined with the prior eager Marlin-vs-tiled parity and qwen Marlin-determinism, the full 2×2 is
green. Perf/capture: **DONE + GO**. No kernel change requested; freeze stands.
