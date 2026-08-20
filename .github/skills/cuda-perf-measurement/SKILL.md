---
name: "cuda-perf-measurement"
description: "Which tool to reach for when measuring CUDA decode/prefill performance in this repo, and the three CUDA-specific traps that have each produced a confidently wrong answer here. Load before profiling a kernel, comparing a GEMV variant, or quoting a tok/s number from the native CUDA backend."
domain: "performance"
confidence: "high"
source: "earned (#1573 wall-clock-vs-nsys inversion, #1574 roofline diagnosis, GEMV bandwidth probe)"
---

# CUDA performance measurement

## Context

`measurement-discipline` covers whether a number means what you think. This skill
covers the layer under it: **which instrument to use on the CUDA backend, and the
device-specific ways each one lies.** Every trap below was hit for real, and each
one produced an answer that was not merely imprecise but backwards.

Read this before you profile a kernel, A/B a GEMV lever, or quote tok/s.

## Pick the cheapest instrument that can answer the question

| Question | Tool | Cost |
| --- | --- | --- |
| Is this kernel bandwidth-bound, and how far from peak? | `decode_gemv_achieved_bandwidth_by_projection_shape` | ~20 s |
| Did a kernel change help, in kernel time? | same probe, before/after | ~40 s |
| Where does a whole decode step go? | `nsys` + marginal differencing (below) | ~10 min |
| Did end-to-end decode change? | `profile_native` | ~1 min/run |
| Does the M>1 verify path cliff? | `marlin_bench` | minutes |

Start at the top. The GEMV probe exists specifically so that iterating on the
int4 GEMV does not require the nsys ritual, which is slow **and** has three ways
to silently produce nonsense.

```bash
# GEMV achieved bandwidth at the real 30B projection shapes.
CUDA_VISIBLE_DEVICES=<idle gpu> ONNX_GENAI_CUDA_DEVICE=0 \
  cargo test -p onnx-runtime-ep-cuda --lib --features cuda \
  decode_gemv_achieved_bandwidth -- --ignored --nocapture
```

It prints per-shape GB/s and, on the last line, the decode tok/s ceiling implied
by GEMV time alone. That ceiling is an upper bound on end-to-end decode, so:

- **ceiling far above measured tok/s** → the loss is outside the GEMVs (launch
  gaps, casts, attention, norms). Do not tune the GEMV.
- **ceiling near measured tok/s** → the GEMVs are the whole story.

Perf probes in this repo are `#[ignore]`d tests, not binaries, so they can reach
crate-private kernel structs. `cargo test ... -- --ignored --nocapture` is the
entry point, and `--list` will show you the rest.

## Trap 1: nsys hides everything inside a CUDA graph by default

**This is the big one.** Without `--cuda-graph-trace=node`, every kernel launched
from a captured graph is **absent** from `CUPTI_ACTIVITY_KIND_KERNEL`. You do not
get a warning, an empty row, or a partial result — the 52-layer transformer
simply is not in the trace.

What that produced here: a report that the GPU was idle 96.4% of the time, that a
token cost 0.11 ms of kernel time, that a 52-layer model issued 2 rmsnorms per
token, and therefore that the transformer was not running on the GPU at all. The
follow-on conclusion — that VMM and memcpy were the decode bottleneck — was
entirely an artifact.

What broke the spell was `nvidia-smi`: 92% utilization, 24 GB resident. **When an
analysis contradicts the most naive possible observation, suspect the
instrument first.**

```bash
nsys profile -t cuda --cuda-graph-trace=node -o out --force-overwrite true <bin>
nsys export -t sqlite -o out.sqlite --force-overwrite true out.nsys-rep
```

Sanity check every trace before interpreting it: a 52-layer model must show
kernel counts that are multiples of ~52. If rmsnorm appears twice per token, you
are looking at a graph you failed to expand.

## Trap 2: load-phase cost masquerades as per-token cost

Kernel counts and API counts include everything since process start. Weight
repacking, prefill, and one-time VMM mapping all land in the same totals as
steady-state decode.

This produced a claim that ~8000 `cuMemCreate`/`cuMemSetAccess` calls (~4.2 s)
were a decode bottleneck. They were identical at 32 and at 96 generated tokens —
7998 vs 8000 — so they were load cost, not per-token cost. Same for
`matmul_nbits_dequant_f16` (416 either way) and `marlin_repack` (1).

**Always difference two runs** at different token counts and divide by the
difference. A single profile cannot separate fixed from marginal cost.

## Trap 3: wall clock on this box cannot resolve anything under ~10%

Run-to-run drift is ±5-10%, and the machine is shared.

- `ONNX_GENAI_GEMV_WIDELOAD` set to `0` and set to `1` produced the *same*
  ~4.7% "improvement" over baseline. The lever did nothing; the drift was real.
- `ONNX_GENAI_INTERLEAVE_DEQUANT` looked like +4% on wall clock while nsys kernel
  time showed it 2% **slower** — and it later turned out the path could not even
  execute on this model. Merging on wall clock alone would have shipped a
  regression as a win.

**Rule:** a perf claim needs kernel time (probe or nsys) or launch counts. Wall
clock is corroboration only, and must be reported with its spread and n.

Also verify the GPU is idle **before each run**, not once per sweep, and pin to a
specific device: `CUDA_VISIBLE_DEVICES=<n> ONNX_GENAI_CUDA_DEVICE=0`.

## Know your roofline, and what a good fraction actually is

Decode is weight-bandwidth-bound: per token you must read every weight once.

```
roofline_tok_s = HBM_peak_bytes_per_s / weight_bytes_per_token
```

For int4 with fp16 scales and int4 zero points, bytes per node are
`N*K*(1/2 + 2/block + 0.5/block)` — at `block=32` that is `N*K*0.578`.

**Do not treat 100% of roofline as the target.** NVIDIA's own published
end-to-end inference data (NVlabs/cutile-rs `sec5_exp3`) puts the best engines at
**0.62-0.67 of nominal roofline** on bf16: grout 0.669, vLLM 0.646, SGLang 0.637.
Those are the numbers to beat, and they were achieved with *no dequantization in
the inner loop at all*, so an int4 path reaching them is doing strictly more work
per byte.

## Environment facts for this box

- `nsys` is not on `PATH`; use `/usr/local/cuda/bin/nsys`.
- **`ncu` is not permitted here.** Occupancy and stall-reason arguments cannot be
  verified directly — treat any comment in the codebase justifying a choice by
  occupancy as an untested hypothesis. One such comment gated split-K off for
  large-N shapes and cost ~5% of decode until it was measured (#1573).
- The GEMV probe needs a dedicated idle GPU; it is `#[ignore]`d for that reason.
