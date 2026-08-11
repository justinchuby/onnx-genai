# Megakernel Phase 0 — Persistent Single-Op QMoE Decode Kernel

**Author:** Quaid (CUDA kernel engineer)
**Date:** 2026-08-11
**Branch:** `squad/megakernel-phase0-qmoe` (off origin/main, #766 merged)
**GPU:** CUDA_VISIBLE_DEVICES=4 (H100/Hopper, 132 SM)
**Verdict:** ⛔ **NO-SHIP** — oracle holds byte-exact, but decode **regresses +7.4%**. Occupancy is the trap, exactly as predicted.
**Phase 1 recommendation:** **Do NOT greenlight Phase 1 as specified.** See bottom.

---

## 1. What I built

Replaced the QMoE **rows==1 decode** chain's two heaviest launches with ONE persistent
fused kernel, keeping the FC1/SwiGLU intermediate (`activated`) in the kernel and
eliminating its global scratch round-trip, using a **counter-based (global-atomic)
producer/consumer** for the FC1→FC2 chunk dependency (the Hazy "4-chunk MLP" trick) —
**NOT** a `grid.sync()` cooperative barrier (per Roper §4: grid barrier risks co-residency
deadlock and fights occupancy).

Important correction to the feasibility spec: the "5-launch / 4-scratch" description in
`roper-megakernel-feasibility.md` §3 is **stale**. Current origin/main (#766) already fuses
FC1+FC3+activate into `qmoe_gate_up_activate`. The **real** decode chain is 4 launches /
2 scratch:

```
route  ->  gate_up_activate (FC1+FC3+SwiGLU)  ->  FC2 (qmoe_linear)  ->  combine
                    |-- activated (16 KB) --|          |-- route_output (64 KB) --|
```

Phase 0 fused `gate_up_activate` + `FC2` into one persistent kernel producing
`route_output`; `combine` stays separate (it sums across routes → cross-block reduction,
must stay separate to keep byte-exact accumulation order).

Design:
- Fixed grid of co-resident blocks (occupancy-capped for deadlock-freedom), cached in an
  `AtomicU32` so warmup == capture geometry.
- **Phase A producers** compute `activated` with accumulation *identical* to
  `gate_up_activate` (same `qmoe_int4_chunk`/`block_sum`/`swiglu_value`, 256-thread stride),
  then `atomicAdd` a per-route done-counter.
- **Phase B consumers** spin on the per-route counter (volatile read + `__threadfence`),
  then compute FC2 *identical* to `qmoe_linear`.
- Tiny `qmoe_fused_reset` kernel zeroes counters each launch (capturable).
- Env toggle `ONNX_GENAI_QMOE_PERSISTENT` (default on) → clean A/B on one binary.
- DRY: driven off tensor shapes/attrs only, **no model-name gates**; multi-row
  (prefill/grouped) path untouched, falls back to per-op for rows>1.

Files: `crates/onnx-runtime-ep-cuda/src/kernels/qmoe.rs` (kernel + launch + toggle),
`crates/onnx-runtime-ep-cuda/tests/qmoe_gpu.rs` (3 new A/B + CPU tests).

---

## 2. Gate 1 — Oracle byte-exact: ✅ HOLDS

35B-A3B teacher-forced fp32 oracle, persistent **ON**:

```
QUAID-PHASE0 QMoE teacher-forced margin = 0.09375 (argmax=33803)
```

- **argmax = 33803**, **margin logprob(33803)−logprob(5342) = 0.09375 EXACTLY.** ✔
- Behavior is **byte-identical** between persistent ON and OFF: same #722 note
  (autoregressive token@119 = 46283 = C1_CAPTURE_TOKEN, the documented benign fp16
  coin-flip — informational, not a fail), same everything.
- Unit tests: **32 pass** (29 existing + 3 new). New tests assert on `to_bits()` →
  the persistent path is **bit-identical** to the per-op path AND matches CPU.

**Environmental note (NOT a Phase-0 failure):** the oracle test's *step-3 DENSE int4
cross-check* aborts with a pre-existing **cuDNN** error
(`CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH` at node 80 `mlp/gate/Softmax`, a Float16
Softmax op on a *different* model). This reproduces **identically with persistent OFF** and
has nothing to do with the QMoE kernel (my kernel touches neither cuDNN nor Softmax). The
QMoE **primary lock (step 1)** and **#722 tripwire (step 2)** both pass byte-identically
with the fused kernel. The correctness gate — the teacher-forced QMoE margin — is GREEN.

This matches my int4/DP4A prior: reduction restructures that preserve per-thread fp32
accumulation order hold the oracle bit-exact. **Numerics are low-risk.**

---

## 3. Gate 2 — Wall-clock ≥3% decode win: ❌ FAILS (regresses)

Steady-state decode, `profile_native --pipeline --steady --warmups 1 --runs 3 --tokens 128`,
`ONNX_GENAI_CUDA_GRAPH=1 ONNX_GENAI_CUDA_KV_MAX_LEN=262144`, GPU 4, median of 3:

| Config | decode ms/tok | throughput | Δ vs baseline |
|---|---|---|---|
| **Baseline** (persistent OFF, per-op) | **11.109** | 90.01 tok/s | — |
| **Phase 0** (persistent ON, fused) | **11.932** | 83.81 tok/s | **+7.4% SLOWER** |

Runs were tight (±0.02 ms across 3), so this is a real **regression**, not noise. The
≥3%-win gate required ≤10.77 ms/tok; we got 11.93. Decisive miss.

---

## 4. Why — occupancy collapse (the mechanism)

Launch geometry, decode (routes=8, inter=512, hidden=2048):

| Kernel | resident blocks | note |
|---|---|---|
| per-op `gate_up_activate` | **4096** | one block per (route, inter-feature), uncapped (routes≤16) |
| per-op `FC2` (`qmoe_linear`) | **2112** | capped at min(tasks, 132 SM × 16) |
| **Phase 0 persistent fused** | **660** | `occupancy_max_active_blocks=5/SM × 132 SM`, **register-limited** |

```
QUAID-PHASE0 persistent_grid: per_sm=5 sm_count=132 grid=660 block_dim=256 smem=1024
```

Fusing FC1-decode + FC2 into one kernel makes it register-heavy → only **5 blocks/SM**
(1280 of 2048 max threads/SM ≈ 62% occupancy). The bandwidth-bound **FC2 GEMV** therefore
runs with **~3.2× fewer** concurrent blocks (2112 → 660) driving HBM. Decode QMoE is
**HBM-bandwidth-bound** (~503 MB/token weight reads); fewer concurrent blocks → lower
memory-level parallelism → lower achieved bandwidth → the op slows.

The thing Phase 0 *saved* — the `activated` scratch round-trip (~16 KB/token, plus the two
launch/drain seams) — is **negligible** vs 503 MB/token of weight traffic. This is fully
consistent with Cohaagen's FC2+combine fusion measuring only **+0.08%** from removing a
seam: at decode the QMoE cost is weight bandwidth, not launch overhead or scratch DRAM.

**One-line:** the fusion that removes the seams also destroys the occupancy the
bandwidth-bound GEMVs need to saturate HBM, and the seams were never the bottleneck.

---

## 5. Verdict & Phase 1 recommendation

**NO-SHIP.** Oracle byte-exact (0.09375, 33803) — the persistent counter-synced structure
is numerically safe and bit-identical to per-op. But it **regresses decode 7.4%** because
decode QMoE is occupancy/bandwidth-bound, not seam-bound. Reverting the branch code; keeping
this memo.

**Phase 1 (the real 1.3–1.6× prize) — recommend DO NOT greenlight as specified.** Phase 0
is the cheapest instance of the whole-step-megakernel thesis, and it inverts: naive fusion
trades occupancy for seam-removal on the heaviest decode op, and loses. Phases 1–2 (fusing
attention + all MoE across the whole decoder step into one persistent kernel) face the SAME
occupancy-vs-fusion tension at **larger** scale — more weights + more state in one kernel →
even higher register/smem pressure → even lower occupancy — while the payoff (seam removal)
stays small because decode is bandwidth-bound. A megakernel could only win here if it holds
**≥ current per-op occupancy**, which requires a real **tile scheduler** (MPK/Hazy-style:
many small tiles dynamically load-balanced across a persistent grid, NOT one-block-per-task
fusion). That is a much larger, still-speculative effort. My recommendation: shelve the
whole-step megakernel; the higher-EV decode lever remains **weight bandwidth** (int4 layout /
DP4A / vectorized weight loads), not launch/seam fusion.

---

## Reproduce

```bash
source .cudaenv.sh   # CUDA_VISIBLE_DEVICES=4, LD_LIBRARY_PATH=/home/tlwu/cudnn9.19_cuda13/lib
# A/B wall-clock:
ONNX_GENAI_CUDA_GRAPH=1 ONNX_GENAI_CUDA_KV_MAX_LEN=262144 ONNX_GENAI_QMOE_PERSISTENT={0,1} \
  ./target/release/profile_native --model /home/justinchu/qwen36-35b-a3b-qmoe-artifacts \
  --pipeline --ep cuda --steady --warmups 1 --runs 3 --tokens 128
# Oracle (primary lock = teacher-forced QMoE margin; dense step-3 hits unrelated cuDNN env error):
ONNX_GENAI_CUDA_GRAPH=1 ONNX_GENAI_CUDA_KV_MAX_LEN=4096 ONNX_GENAI_QMOE_PERSISTENT=1 \
  cargo test -q -p onnx-genai-engine --features "cuda native-backend" \
  --test qwen36_35b_a3b_qmoe_divergence \
  qwen36_35b_a3b_qmoe_native_cuda_matches_fp32_oracle -- --ignored --nocapture
```
