# Decision: shuffle-fusion (split+transpose) — HONEST FLOOR, no code shipped

- **Agent:** Gaff (CUDA-kernel specialist)
- **Branch:** `squad/shuffle-fusion`
- **Date:** 2026-08-19
- **Verdict:** ❌ **Do not ship.** The dominant decode "data-shuffle" kernels
  (60 transpose + split) are already amortized to near-free *inside the
  captured CUDA decode graph*. Eliminating them via zero-copy views does not
  improve end-to-end tok/s (within-noise) and structurally **regresses** the
  capture by fragmenting the single graph into eager seams. This is a verified
  practical floor for the captured decode path, not a tuning miss.

## What was attempted

Deckard attributed ~289µs/step of "redundant memory movement" (187 standalone
byte-copy kernels/step: 60 transpose + 66 split + 37 gather + 24 concat) via
nsys kernel counts, and estimated +10–20 tok/s from fusing the bulk
(split+transpose = 126 kernels).

Chosen approach (least-invasive, generality-preserving): the executor already
has an EP-agnostic zero-copy `Kernel::view_outputs()` hook (CPU EP uses it for
Transpose/Reshape/Slice; the **CUDA EP used it nowhere**). In decode `seq_len=1`,
every one of these transposes is a **pure layout no-op** — the permutation only
reorders unit-extent axes, so the output is byte-for-byte the input with a new
shape. That makes them eligible to be delivered as an aliasing contiguous view
instead of a copy kernel. Implemented `view_outputs` on the CUDA `TransposeKernel`
(no-op detection only; genuine transposes fall back to the copy kernel), gated by
`ONNX_GENAI_CUDA_DISABLE_SHUFFLE_FUSION=1`.

## Why it does not work — the capture boundary

A zero-copy view install is a **host-side** operation: it drops the output's own
device buffer, records an aliasing entry in `views_meta`, and pins the source.
Those buffer-lifetime mutations are **illegal inside an active CUDA-graph capture
region**. Observed directly:

- With views ON, the capture recording pass **aborts device-graph recording** on
  the first transpose view, so the executor's quarantine loop (run.rs) forces the
  **entire `Transpose` op-type to eager seams** and re-plans.
- Segment count (`ONNX_GENAI_LOG_CAPTURE_SEGMENTS=1`), qwen3.5-0.8b fp16io, H200:
  - views **OFF** (baseline): **16 captured segments / 15 eager seams**
  - views **ON**: **70 captured segments / 69 eager seams**
    (seam breakdown: 120 Transpose [device-recording-aborted → quarantined],
     plus the pre-existing Reshape/Range/Squeeze seams).

I confirmed the elision path itself was firing correctly (960 no-op detections,
zero declines, bit-exact contiguous views) and even made the view nodes report
`capture_support = Supported` — but that is upstream of the real gate: the view
*install* aborts recording, so the op-type is quarantined regardless. The
zero-copy view mechanism is fundamentally an **eager-path** optimization
(the `decode_view_plan` reinstates aliases on replay *outside* the graph); it
cannot live inside a captured segment.

## Why the ceiling is real (not just this approach)

Even a perfect, capture-safe removal of these transposes (e.g. a build-time graph
rewrite folding the no-op transpose into consumer index math) would recover
almost nothing:

- These are `seq_len=1` layout no-ops → each is a ~`[B,1,D]` fp16 memcpy
  (~2 KB). 60/step at H200 ~3 TB/s ≈ **~40 ns/step total**.
- A decode step is ~**4.0–4.2 ms** (245 tok/s). The transpose copies are
  ~0.001% of the step. They replay near-free inside the captured graph.

Deckard's 289µs/step "redundant memory movement" is therefore an **eager-mode /
nsys standalone-kernel-count artifact** — it counts kernel *launches*, which the
captured graph has already amortized to ~zero launch overhead. The count is real;
the *cost* on the captured path is not.

## Measurements (H200, GPU 0, qwen3.5-0.8b fp16io, `--tokens 128 --runs 2`)

Paired, interleaved A/B (views ON default vs `ONNX_GENAI_CUDA_DISABLE_SHUFFLE_FUSION=1`):

| pair | ON (tok/s) | OFF (tok/s) |
|------|-----------|-------------|
| 1    | 243.1     | 238.4       |
| 2    | 245.9     | 250.1       |
| 3    | 245.3     | 246.4       |

The ON/OFF gap (~245 vs ~245) is **smaller than the within-arm spread**
(OFF alone ranged 238–250). Per the profiling skill's noise discipline, this is
**unmeasured** — there is no effect to report beyond noise, and the structural
capture regression (70 vs 16 segments) makes the ON path strictly worse in
robustness. No win.

## Decision

- **Reverted** the `movement.rs` `view_outputs` change — origin/main behavior is
  unchanged. No opt-out flag shipped (a default-OFF flag guarding a
  capture-fragmenting path is pure maintenance cost for zero benefit).
- **Did not attempt** the concat/gather bucket: those 24/37 are overwhelmingly
  SHAPE-graph scalar ops (Range/Reshape/Squeeze on shape tensors), not real data
  movement, and the same capture-boundary logic applies to any view-based fusion.
- The genuinely remaining decode gap vs ORT-eager is **not** glue-kernel launch
  overhead on the captured path; future attribution should measure the
  **captured-replay** cost (ncu `--graph-profiling node` / nsys
  `--cuda-graph-trace=node`), not standalone eager kernel counts, before
  scoping a "fusion" lever.

## Reusable finding (stored to memory)

> The CUDA captured decode graph already amortizes tiny movement kernels
> (transpose/split/concat/gather) to near-zero. Zero-copy `view_outputs` elision
> of them is an EAGER-path optimization: a view install mutates device-buffer
> lifetimes, which is illegal mid-capture, so it aborts graph recording and
> quarantines the op-type to eager seams — fragmenting the single decode graph
> (e.g. 16 → 70 segments) and regressing/neutralizing tok/s. Attribute
> movement-kernel cost on the captured-replay path, not by eager standalone
> kernel counts.
