# Decision drop — Leon: captured spec-decode byte-identity vs speedup

**Owner:** Leon (CUDA/Rust kernel engineer)
**Branch:** `squad/spec-decode-fp32-verify` (PR #1006, DRAFT)
**Base:** `aceed718` (Batty workspace-width-leak / miss-path fix)
**Date:** 2026-08-15

## TL;DR

Byte-identity to plain M=1 greedy is **achievable and capture-safe** (qwen 5/5;
glm 3/5 with the remaining 2 being an *attention*-path reassociation, not the
GEMV). But **G2 (spec faster than greedy) is fundamentally unachievable together
with byte-identity at the acceptance rates prompt-lookup produces on normal
prompts** — and it is *not a GEMM/kernel problem*: even the fast, non-byte-
identical Marlin verify loses to greedy at these acceptance rates. This is the
evidence-backed "STOP and report" case the mission described. I did **not** ship
a hollow "green" PR; #1006 stays DRAFT with this diagnosis.

## Phase-1 diagnosis (confirmed)

The M=W verify diverges from M=1 greedy because the batched int4 GEMM
**reassociates the K-reduction** (Marlin fp16-accumulate tensor cores, and the
portable tiled fp32 GEMM) relative to the M=1 fp32/fp16 GEMV. It is a genuine
argmax flip, not a near-tie (near_tie=0 at every divergence). Also confirmed the
prior reference was non-deterministic (Marlin-prefill-ON greedy); the correct
reference is **Marlin-OFF greedy**.

## What I built (works, committed)

Single-launch **batched byte-identical captured-verify GEMV** kernels, one per
M=1 reduction family, routing each M=W verify GEMM to the batched sibling of the
exact kernel greedy uses; each row's K-reduction is bit-identical to M=1 *by
construction*:
- General single-warp (q / o / lm_head) — committed `58cd1a89`.
- Tall-skinny **down-projection** (per-block float accumulate + warp/8-warp combine).
- Fused **gate_up SwiGLU**, both plain and the **rmsnorm-prologue** form
  qwen2.5-14b uses (normalized activation recomputed on-the-fly → no M×K shared,
  capture-safe, byte-identical).
- Fixed `run_f16_per_row_verify` to dispatch gate_up nodes correctly.

All advertise `last_call_capture_safe` → the verify graph captures **whole-graph
(segments=1)**, zero capture-time device alloc.

Opt-in guards retained: `ONNX_GENAI_SPEC_BATCHED_VERIFY=1` (fast byte-identical
path), `ONNX_GENAI_SPEC_PERROW_VERIFY=1` (slow per-row oracle reference).

## Gate results

| Gate | Result |
|------|--------|
| **G1 byte-identity** | qwen **5/5** (W=5..9, accepted 95–104, near_tie=0). glm **3/5** (W=5,7,8 identical; W=6@111, W=9@157 diverge in the **attention** path — q_seq=W flash-attn vs q_seq=1 gqa_decode reassociation, *not* the GEMV). |
| **G2 speedup** | **FAIL — fundamentally, for any kernel.** See below. |
| **G3 capture** | **PASS** — whole-graph (segments=1), captures>0, fallbacks=0, zero capture-time alloc. |
| **G4 hygiene** | **PASS** — f64 oracle 25/0; fmt clean; only the pre-existing allowed `platform_capacity.rs:247/249` clippy casts. |

### Byte-identity matrix (batched, W×model)

| W | qwen | glm |
|---|------|-----|
| 5 | ✅ | ✅ |
| 6 | ✅ | ❌ (attn @111) |
| 7 | ✅ | ✅ |
| 8 | ✅ | ✅ |
| 9 | ✅ | ❌ (attn @157) |

## Why G2 is unwinnable at realistic acceptance (DECISIVE evidence)

`leverb_phase0` probe, qwen2.5-14b, **whole-graph captured replay walls**
(segments=1, zero capture-time alloc):

| Path | M=1 | M=8 | ratio |
|------|-----|-----|-------|
| M=1 greedy | 7.9 ms | — | 1.0x |
| **Marlin M=8** (fp16 TC, *not* byte-identical) | — | **35.9 ms** | **4.5x** |
| **Batched byte-identical M=8** | — | **65 ms** | **8.3x** |
| Portable tiled M=8 (not capture-safe → 96 segments) | — | 123 ms | 15.6x |

An M=W verify forward is inherently **~4.5x (Marlin) / ~8x (byte-identical)** an
M=1 step on this model/HW — attention (M queries), the 152064-vocab lm_head, and
the MLP all scale ~M. "Captured M=W verify ≈ M=1 cost" (the weights-read-once
premise) is **false for any kernel** here; the GEMV is compute/occupancy-bound at
these sizes, not weight-bandwidth-bound, so reading weights once does not recover
M=1 cost.

**Break-even** (no thrash): spec beats greedy only when
`verify_ms / mean_accept < 7.9 ms`:
- Marlin: **mean_accept > 4.5**
- Byte-identical: **mean_accept > 8.2**

**Measured mean_accept** (prompt-lookup, normal + repetitive prompts): **2.4–3.3**.
→ Marlin spec ≈ 0.72x, byte-identical spec ≈ 0.40x *even with zero thrash*.
The existing (baseline) Marlin captured spec is **also slower than greedy** here
— so "preserve the speedup" is vacuous in this harness/config: there is no
speedup to preserve at prompt-lookup acceptance rates.

### Second-order cost: verify-graph re-warm THRASH

The verify graph and the M=1 decode graph share the **single EP device-graph
slot**; miss-path M=1 decodes call `reset_device_graph()`, evicting the captured
verify graph and forcing a re-warm + re-capture (2 non-replay steps + a full
whole-graph capture each). End-to-end this pushes spec to **0.04–0.21x** (vs the
~0.4–0.7x no-thrash floor) and produces large run-to-run variance
(e.g. W=9: 9.5s ↔ 32s ↔ 41s across runs). This affects **all** verify kernels
equally; it is orthogonal to byte-identity.

## Recommendation (contract relaxation)

The contract "captured spec MUST be byte-identical to greedy **AND** faster than
greedy" is internally contradictory at achievable acceptance on this model/HW.
Recommend splitting it:

1. **Ship byte-identity as an opt-in determinism/reproducibility mode**, not a
   speed feature. `ONNX_GENAI_SPEC_BATCHED_VERIFY` gives byte-identical,
   capture-safe spec-decode (~8x M=1/verify) for reproducible/debuggable runs.
   The kernels are correct and whole-graph capturable today (qwen 5/5).
2. **Drop the "spec beats greedy on prompt-lookup" requirement** — it cannot,
   for *any* verify kernel, at mean_accept < 4.5. Re-scope the speed win to
   regimes that actually reach mean_accept > 4.5–8 (a real draft model / tree
   drafting, or batch > 1), and validate G2 there.
3. **Highest-leverage speed fix (independent of byte-identity):** give the
   captured verify its **own** device-graph slot so miss-path M=1 decodes don't
   evict it. This removes the 2–6x thrash penalty for *all* verify kernels. It
   still won't make byte-identical spec beat greedy at mean_accept < 8, but it is
   the correct fix for the harness and unblocks a fair G2 measurement at high
   acceptance.
4. If byte-identity at high acceptance is desired, the glm attention divergence
   (W=6,9) needs an **fp32 verify-attention** (q_seq=W kernel using the same
   per-query online-softmax reduction as q_seq=1 gqa_decode) — tractable, same
   per-row principle as the GEMV, but not built here because G2 is blocked
   regardless.

## Repro

```
source /home/justinchu/onnx-genai/.cudaenv.sh
# byte-identity (batched): qwen 5/5, glm 3/5
CUDA_VISIBLE_DEVICES=<idle> ONNX_GENAI_RUN_CUDA_SMOKE=1 ONNX_GENAI_SPEC_CAPTURED_VERIFY=1 \
  ONNX_GENAI_SPEC_GATE=0 ONNX_GENAI_SPEC_BATCHED_VERIFY=1 \
  cargo test -p onnx-genai-engine --features cuda,native-backend --test native_speculative_driver \
  leon_gate1_qwen_batched_verify_cuda -- --nocapture
# decisive captured replay walls (batched vs Marlin vs portable):
CUDA_VISIBLE_DEVICES=<idle> ONNX_GENAI_RUN_CUDA_SMOKE=1 ONNX_GENAI_LEVERB_BATCHED=1 \
  ONNX_GENAI_LEVERB_MODEL=/home/justinchu/shared-models/qwen2.5-14b-instruct-int4-zp-onnx \
  cargo test -p onnx-genai-engine --features cuda,native-backend --lib \
  leverb_phase0_capture_probe -- --nocapture --ignored
```
