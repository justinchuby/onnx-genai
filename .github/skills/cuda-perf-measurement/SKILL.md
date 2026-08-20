---
name: "cuda-perf-measurement"
description: "Which tool to reach for when measuring CUDA decode/prefill performance in this repo, and the three CUDA-specific traps that have each produced a confidently wrong answer here. Load before profiling a kernel, comparing a GEMV variant, or quoting a tok/s number from the native CUDA backend."
domain: "performance"
confidence: "high"
source: "earned (#1573 wall-clock-vs-nsys inversion, #1574 roofline diagnosis, #1581 probe that measured its own dispatch)"
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

## Trap 4: a microbenchmark that times one launch is timing the host

`cuEventRecord` timestamps the *stream*, so `record(start); kernel.run(); record(end)`
encloses the host-side dispatch as well as the kernel. In this repo `run()` costs
about **24 us** — it re-reads tuning env vars, re-derives the launch geometry and
re-looks-up the NVRTC entry every call. That is larger than several of the decode
kernels themselves.

What that produced: the GEMV probe reported the GQA k/v projection at **28 GB/s**,
1.4% of peak, and an issue was filed against a "starved grid". The kernel's real
time was **4.6 us, 214 GB/s**. Five sixths of the reading was the host.

**Rule:** enqueue a *batch* between the events, and prefer a **captured CUDA
graph** for the timed region. Graph replay is also what production decode does, so
the number is representative as well as correct. Time the host separately and
print it — an uncaptured path really is limited by it.

## Trap 5: an idle GPU is clocked down, and it ramps slower than your probe runs

An idle A100 in this box sits at **210 MHz against a 1410 MHz maximum**, with
persistence mode off. A probe that starts timing immediately reports whichever
partial ramp it caught.

Measured cost: **17%** on the widest projection. Worse, the *direction* of the
bias depended on whether a neighbour happened to be running work on an adjacent
GPU — the same code measured 816 GB/s on a warm device and 586 GB/s on a cold
one. Any A/B split across those two states is noise wearing a result's clothes.

**Rule:** ramp before measuring, and prove the device held still.

- Ramp until the timing stops improving **and** a wall-clock floor has elapsed
  (~8 s of continuous work). Convergence alone is not enough: two short readings
  agreed at "107 -> 107 us" on a device that then measured 90 us.
- Re-measure the *first* shape at the *end* of the sweep and report the drift. If
  it moved more than a few percent, the rows were measured under different
  conditions and cannot be ranked against each other.
- `nvidia-smi --query-gpu=index,clocks.sm,power.draw --format=csv` while the probe
  runs is the cheap sanity check. Low power draw (an A100 near 70 W of a 400 W
  TDP) means the device is mostly idle between your launches.

## Sanity-check every absolute number against something independent

The two traps above were caught by the same observation, not by suspicion: in a
sweep over shapes, **per-shape time must be ordered by bytes moved**. When the
1 MB k/v projection measured *slower* than the 16 MB q projection, no bandwidth
story could explain it — which meant the instrument was wrong, not the kernel.

Build that kind of internal cross-check into any probe you add. A microbenchmark
with no self-contradicting case is a microbenchmark you cannot debug.

## Trap 6: an optimization that ships, is tested, and never runs

The device argmax was implemented, GPU-tested, benchmarked at +8%, and made
"default on" -- and then ran for months on exactly zero real requests. Two
independent gates, each innocuous on its own, closed the path:

1. `PipelineDecodeLoopBackend` never overrode `greedy_fastpath_supported()`, so
   it inherited the trait default of `false`. Nothing failed; the pipeline just
   read 404 KB of logits per token forever.
2. The eligibility test was `chain.is_empty()`, but every real model ships
   `top_k` / `top_p` in its `generation` defaults and every chat template adds a
   stop sequence. None of those can move an argmax. The tests all used bare
   `GenerateOptions::default()`, so the chain was empty in tests and never in
   production.

**A default-`false` capability query and a test that only exercises the default
config are a matched pair that will hide any fast path.** When a lever is
supposed to be on, prove it fires on the real model -- `--profile` and look for
the span that should have *disappeared* (`loop.sampling`), not just at tok/s,
which was inside the noise band here.

The second gate had a second lesson. `chain.is_empty()` was not merely too
strict, it was the wrong question: a greedy request should not have been
carrying sampling warpers at all. When a fast path needs an exemption to fire on
ordinary input, check whether the input should exist before you write the
exemption.

And this pair of fixes is where Trap 5 bit hardest: the new code measured 36.7
tok/s against a 33.9 tok/s baseline taken twenty minutes earlier, which looked
like the +8% the archive predicted. Re-measuring the baseline **back to back**
gave 36.5. The entire "win" was the A100 ramping off its 210 MHz idle clock.
**A baseline from earlier in the session is not a baseline.**

## Trap 7: profiling a binary that does not contain the code you changed

The CLI has two CUDA feature paths and only one of them is this repo's hand-written
backend:

| Feature | What you get |
| --- | --- |
| `--features native-cuda` | **this repo's CUDA EP** (`onnx-runtime-ep-cuda`) |
| `--features cuda,native-backend` | ORT's CUDA EP; `onnx-runtime-ep-cuda` is **not in the dependency graph at all** |

Building with the second and editing a kernel produces a `cargo build` that
finishes in under a second and changes nothing, because the crate you edited is
not a dependency. `cargo tree -p onnx-genai-cli -i onnx-runtime-ep-cuda` printing
"nothing to print" is the tell. The measured effect was not subtle -- p50 per-token
went from 24 ms to 532 ms -- but it reads as "my change caused a 22x regression"
rather than "I am running a different backend", and the instinct is to go debug
the change.

**Before quoting any number from a rebuilt binary, prove the binary contains the
build:**

```bash
strings -a target/release/onnx-genai | grep -c '<a string only your new code has>'
```

Zero means you measured the old binary or the wrong feature set. This costs one
second and removes an entire category of confidently wrong answers.

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
