# Native-vs-ORT A/B harness

Generators and an interleaved driver for comparing this repo's CPU execution
provider against a **real ONNX Runtime CPU session** on the same host, the same
graph, the same thread count and the same inputs.

These are the scripts behind
[`docs/benchmarks/2026-08-15-cpu-ep-vs-ort-attention-moe.md`](../../docs/benchmarks/2026-08-15-cpu-ep-vs-ort-attention-moe.md).

## Why this exists

A kernel microbenchmark that compares a new kernel against the *old* kernel
answers "did I make our code faster", which is not the question that matters.
The question that matters is "is our execution provider faster than the runtime
a user would otherwise run". Those two answers can point in opposite
directions: a kernel can get 8× faster and still be 3× slower than ORT.

So every number here is a **native/ORT ratio measured inside one process, on one
host, on the same graph**, with node and session overhead included. Lower is
better; `1.0` is parity.

## Ground rules

* **Ratios, not absolutes.** Shared CI/dev hosts drift. Same-shape absolute
  timings on the reference host moved by more than 4× between sessions, while
  the paired ratio stayed stable. Only publish ratios.
* **Interleaved arms.** `ab.py` alternates the arms trial by trial, so drift
  hits both arms roughly equally instead of being attributed to whichever arm
  ran during a noisy minute.
* **Medians and dispersion.** Report `p50` with the observed `[min–max]` of the
  per-trial ratios. A win narrower than the dispersion is not a win.
* **Warmups.** Both runtimes get warmup iterations before the measured runs;
  first-touch page faults and lazy packing otherwise land entirely on whichever
  arm goes first.
* **Parity is checked on every cell.** The driver records the harness's
  `parity=PASS/FAIL`. A performance number from a cell that does not produce
  ORT's answer is not a performance number.
* **Production shapes first.** Head counts, KV-head counts, head dims, hidden
  sizes and expert geometry come from public model configs. Benchmarks are not
  chosen to flatter the kernel.

## Synthetic data

**No trained weights are downloaded or used.** The generators emit single-node
graphs whose *dimensions* come from public architecture configs (Llama-3-8B,
Phi-3-mini-4k, Qwen2.5-0.5B, Qwen3-0.6B, Qwen3-MoE, Mixtral, Phi-3.5-MoE,
BERT-base/large, CLIP, Whisper). Tensor contents are the benchmark harness's
deterministic synthetic pattern, fed identically to both runtimes. Where a full
expert bank would not fit in host memory as f32, the expert **count** is reduced
and the reduction is recorded in the file name (`e{N}`).

This measures kernel and scheduling behaviour at production geometry. It does
**not** measure end-to-end quality, and it cannot detect a data-dependent
performance cliff that only trained weights would trigger.

## Generators

| Script | Emits |
|---|---|
| `gen_gqa.py` | `com.microsoft::GroupQueryAttention`, one node, fully static shapes |
| `gen_grid.py` | the GQA decode/prefill grid across four model geometries |
| `gen_l3sweep.py` | GQA decode graphs whose per-head attended-KV working set lands on 1/2/4/8/16/32 MiB, for cache-topology sweeps |
| `gen_mha.py` | `com.microsoft::MultiHeadAttention` (the operator the vectorised `sdpa_f32` path serves) |
| `gen_moe.py` | `com.microsoft::MoE` / `QMoE`, top-k routing, grouped experts |
| `gen_transforms.py` | the transforms that *surround* attention: `Softmax`, `RotaryEmbedding`, KV-cache `Concat`, BSNH↔BNSH `Transpose` |

Each takes an output directory:

```bash
python3 scripts/ort_ab/gen_transforms.py --out /path/to/models/transforms
python3 scripts/ort_ab/gen_grid.py --out /path/to/models/grid
python3 scripts/ort_ab/gen_moe.py --out-dir /path/to/models/moe --tokens 1 32 512
```

`gen_gqa.py` bakes semantically-constrained integer inputs (`seqlens_k`,
`total_sequence_length`) as **initializers**, because the harness would
otherwise fill them with its generic synthetic integer pattern and both
runtimes would attend over a nonsensical KV length.

## Driver

Build the benchmark binary first (a `cuda-*` feature is required by the crate's
feature wiring even for a CPU-only run):

```bash
cargo build --release -p onnx-genai-bench \
  --no-default-features --features mlas,cuda-13000 --bin bench_generic
```

Then:

```bash
python3 scripts/ort_ab/ab.py \
  --arms base=/path/to/baseline/bench_generic new=./target/release/bench_generic \
  --models /path/to/models/transforms/*.onnx \
  --threads 1 8 16 \
  --trials 5 --runs 7 --warmups 3 \
  --csv results/transforms.csv
```

* `--arms name=path` — one or more binaries. Two arms is the usual case: an
  exact single-commit baseline build and the branch build, so the arms differ
  *only* by the commits under test.
* `--arm-env name=KEY=VALUE` — per-arm environment, for A/B-ing an opt-in
  threshold or feature flag using one binary in both arms.
* `--threads` — passed through as `--native-threads`, which sets
  `ONNX_GENAI_CPU_DECODE_THREADS` **and** confines the process to that many
  CPUs, so the ORT session in the same process is equally constrained. That is
  what makes the thread-count comparison fair.

The driver prints a per-trial line for every cell and a medians table at the
end, and writes the full per-trial CSV at exit.

## Reading a result

```
sm_decode_h32_kv8192  t=8  base: ratio_p50=71.099 [65.886-85.249] native_p50=2.308ms
                            new:  ratio_p50= 6.390 [ 5.973- 7.188] native_p50=0.178ms
```

The publishable claim is "13× closer to ORT, still 6.4× behind at 8 threads" —
not "13× faster". The `native_p50` column is diagnostic only; compare it across
arms within one run, never across sessions.

## Caveats

* The driver **raises** if a cell produces no result line, rather than silently
  dropping it. Under heavy host contention a cell can fail this way; re-run it
  standalone before concluding anything.
* Under `ONNX_GENAI_PROFILE_OPS=1` the GQA fusion decision inverts, so the
  op-level profiler is not trustworthy for measuring scheduling changes. Use it
  to *locate* a hot op, then measure the op in isolation with these graphs.
* `Softmax`/`RotaryEmbedding`/`Transpose`/`Concat` graphs are single-node, so
  fixed per-run session overhead (currently a fresh allocation of every graph
  output) is a large fraction of the smallest cells. Cells below roughly 100 µs
  are overhead-dominated and should be read as an upper bound on the kernel gap,
  not as the kernel gap.
