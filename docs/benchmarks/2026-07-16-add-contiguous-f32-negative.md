# Contiguous f32 Add fast path — negative result (2026-07-16)

## Setup

- Model: `/home/justinchu/qwen2.5-0.5b-int4-onnx`
- Build: `cargo build --release -p onnx-genai-bench --features bench-native`
- Runtime: CPU EP, `RAYON_NUM_THREADS=24`, 24 generated tokens
- Measurement: six order-balanced baseline/candidate pairs; each process used one
  warmup and five measured runs
- Profiling: `ONNX_GENAI_PROFILE_OPS=1`

Every run produced the same first four greedy tokens:
`[11576, 42740, 11, 358]`.

## Candidate

The candidate mirrored the existing `Mul` optimization. Equal-shape contiguous
f32 inputs were borrowed directly and added into the non-aliasing contiguous
output buffer. Broadcasting, strided layouts, non-f32 dtypes, and possible
input/output aliasing retained the generic path. Dedicated tests covered the
fast path and broadcast fallback; all 413 CPU EP tests passed.

## Result

The baseline Add profile share was 2.42% (median across processes).

| metric | baseline median | candidate median | paired median change |
|---|---:|---:|---:|
| Add | 0.4324 ms/step | 0.4238 ms/step | -1.3% |
| Add node share | 2.42% | 2.29% | -0.11 percentage points |
| Decode throughput | 52.85 tok/s | 51.42 tok/s | **-1.5%** |

The host was noisy: an initial sequential sample suggested a larger local Add
win and +5.3% throughput, but the order-balanced pairs did not reproduce it.
Add improved only 0.9–4.8% per pair, while decode throughput regressed in five
of six pairs. The optimization was therefore flat locally and negative
end-to-end, so the source and test changes were reverted.

## Decision

Do not ship a dedicated contiguous-f32 Add fast path on this kernel as measured.
At roughly 2.4% of node time, removing its temporary allocations is below the
current system noise and did not improve decode throughput. Revisit only with a
larger fusion that removes Add dispatch/output traffic together with an adjacent
residual or normalization operation.
