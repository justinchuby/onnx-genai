# Multi-row decode GEMV: ceiling probe kills the large-change hypothesis on the resident model

**Date:** 2026-08-18
**Author:** Copilot (multi-request batching slice)
**Owner directive:** "继续推进 … multi-request batching 对大模型的支持，提高速度，实现简洁高效"
— after the looped single-row decode GEMV removed the M≥2 tiled-GEMM cliff
(PR #1312), is a *true multi-row GEMV* (one weight read, M outputs — the 1/N
amortization applied at the kernel level) worth a large multi-kernel CUDA change?

## The hypothesis under test

The looped GEMV (#1312) removes the sub-crossover batching *penalty* but still
reads the full weight matrix **M times per step** (once per row). A multi-row
GEMV reads each weight **once** and emits M outputs — turning "batch-N is not
penalised" into "batch-N is actually cheaper per token". The owner greenlit
building it **only if a cheap roofline check first shows the prize is there**:
confirm decode at M=2..8 is weight-bandwidth-bound, so collapsing M reads into
one predicts ~M× on the GEMV portion. If it is not bandwidth-bound, stop.

## Hardware / method (house rule §32.2)

- **Box:** RTX 4060 Laptop 8 GB (driver 591.55, CUDA 13.1), Intel i7-13800H
  (14C/20T), `--test-threads=1`. Memory clock 8001 MHz, ~256 GB/s peak.
- **EP:** native CUDA (`--features "bench-native,cuda"`), captured
  `decode_greedy_batch` (graph captures=1, replays=62, fallbacks=0).
- **Model:** `qwen05b-q4` (0.5B int4) — resident, the only model on this box
  whose decode GEMVs run without weight streaming, so the matmul cost is
  observable rather than hidden behind HtoD.
- **Instrument — a control-arm ceiling probe (Nsight unavailable on this box).**
  A temporary, byte-invariant probe (`ONNX_GENAI_DECODE_GEMV_PROBE_ROWS=1`,
  reverted after measuring) caps the looped GEMV to **one** iteration per
  MatMulNBits node instead of M. Output for rows 1..M is discarded (timing
  only), so the step models a *perfect multi-row GEMV that reads weights once*.
  The multi-row change **provably cannot beat** this ceiling, so the delta
  between the real looped step (M reads) and the probe step (1 read) is the
  maximum achievable multi-row prize. Same binary, back-to-back, median of 64
  steps (per-step wall is tail-noisy under WDDM/VMM pressure #863; the median is
  stable and the A/B delta is what matters).

## Result — `qwen05b-q4`, resident, same-binary A/B

| M | real looped ms/step (M reads) | ceiling probe ms/step (1 read) | prize (delta) |
|---|-------------------------------|--------------------------------|---------------|
| 2 | 13.94                         | 12.99                          | **0.95 ms (~7%)** |
| 4 | 15.37                         | 14.77                          | **0.60 ms (~4%)** |
| 8 | 14.97                         | 15.75                          | **−0.78 ms (~0%, at noise floor)** |

At M=8 the "ceiling" is *slower* than the real path — the deltas are at the
same-arm noise floor, exactly the control-arm gate from the profiling skill:
when the effect is comparable to the spread of its own arm, it is unmeasured.

## What this says (measured)

- **The multi-row prize on the resident 0.5B model is ≤1 ms/step (≤7% at M=2,
  ~4% at M=4, ~0% at M=8).** The M≥2 decode step (~14 ms) is dominated by
  **fixed non-matmul batch overhead** (batched attention/GQA, KV append,
  scheduling, per-row sampling, launch), **not** redundant weight reads.
- The ~0.95 ms M=2 delta matches the arithmetic for the redundant read: one
  extra full-weight GEMV pass ≈ 168 matmul nodes × ~2 MB int4 weights /
  ~256 GB/s ≈ 1.3 ms. **The 0.5B weight matrices (~2 MB each) are simply too
  small for the weight read to bind** — the decode GEMV is
  latency/issue-bound, not bandwidth-bound (matches the skill's decode signal).
- Decomposition: at M=1 the whole step is 2.59 ms (matmul ≈0.95 ms ≈37%). At
  M=2 the step is ~14 ms — the **non-matmul** portion exploded 5.4× (≈1.6 ms →
  ≈12 ms) while the matmul portion only doubled. The real small-batch limiter
  on this model is the fixed batch-decode overhead, not MatMulNBits.

## Disposition — do NOT build the multi-row GEMV (for this target)

A large, risky, multi-kernel CUDA change (every GEMV variant × block size ×
zp/bias/rmsnorm) that returns ≤7% at M=2 and ~0% at M=8 on the only resident
model we can measure is precisely the "large refactor that does not move a
measured number" that 简洁高效 forbids. Killed with data, not built.

## Inferred (untestable on this box), for the next agent

On a datacentre GPU where a large model (14B+) fits in VRAM, the weight matrices
are ~10–30 MB each and a resident decode GEMV is far more likely to be
bandwidth-bound, so the multi-row prize *could* be larger there. That regime is
**not measurable on this 8 GB box** (14B streams; HtoD dwarfs the matmul — see
`2026-08-18-batch-n-scaling-8gb-limiters.md`). Two caveats before anyone revives
this: (1) the fixed non-matmul batch overhead that dominates here may persist
and cap the prize regardless of weight size — measure the *fraction*, not just
kernel bandwidth; (2) reuse this ceiling-probe method (cap loop iterations,
diff the step) to size the prize **before** writing kernels.

## Adjacent lead (not chased — flagged for routing)

The real small-batch limiter surfaced here is the **~14 ms fixed batch-decode
overhead** (the 5.4× M=1→M=2 jump in the non-matmul path), independent of the
matmul. That is attention/KV/scheduler territory, a different slice from this
one; noted for the owner to route rather than chased blind.

## Related

- PR #1312 — looped single-row decode GEMV (the merged fix this probe follows).
- #1291 / #1292 / `2026-08-18-batch-n-scaling-8gb-limiters.md` — batch-N scaling
  study and the two 8 GB limiters.
- #1282 — device-sampling producer, the prior data-killed hypothesis.
- #1295 — VMM churn ceiling (streaming-regime limiter A), routed to offload.
