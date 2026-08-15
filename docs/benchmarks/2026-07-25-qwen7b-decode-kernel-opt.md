# Qwen2.5-7B int4 native CUDA decode — down-projection grid-fill (2026-07-25)

## Headline

The Qwen2.5-7B int4 native CUDA decode down projection
(`MatMulNBits` `_scales_f16_down`, symmetric int4, block-32) was **grid-starved**
at 0.57 waves/SM on the H200. Splitting its columns-per-CTA from 8 to 2
(**bit-identical output**, 4× larger grid) lifts steady-decode throughput
**+2.08%**, widening the native-vs-ORT margin on our thinnest model from
**1.100× to 1.121×**.

- Baseline (8 cols/CTA): **301.76 tok/s** (median of 3).
- Optimized (auto → 2 cols/CTA): **308.04 tok/s** (median of 3).
- Over ORT-CUDA (274.75 tok/s): **1.100× → 1.121×** (+10.0% → +12.1%).
- Greedy decode token IDs are **byte-identical** to baseline.

## Where the time goes (nsys kernel summary, decode)

`nsys --cuda-graph-trace=node`, Qwen2.5-7B, CUDA decode. MatMulNBits GEMVs are
~89% of decode GPU time; GroupQueryAttention is only ~9.6% (the per-op timer
inflates GQA because it syncs each of GQA's three sub-kernels per node — do not
trust its absolute attribution).

| Kernel | % decode | avg | note |
|---|---:|---:|---|
| `gate_up_swiglu_rmsnorm` | 42.2% | 48.6µs | fused MLP up, **2.99 waves** — already fills the SMs |
| `scales_f16_down` | 20.3% | 23.3µs | **down proj, 0.57 waves — grid-starved** ← target |
| `scales_f16_rmsnorm` | 19.3% | 22.3µs | qkv proj |
| `scales_f16` | 7.7% | 8.4µs | o proj (0.42 waves, small K) |
| gqa (attention+merge+prep) | 9.6% | — | 3 kernels |

The 42% gate-up kernel is at 2.99 waves — already well-filled, and prior
weight-prefetch attempts on it regressed; it is **not** the lever. The down
projection is the largest grid-starved kernel and the highest-value target.

## Why the down projection was starved, and the fix

The down kernel has a large K (18944) and small N (3584). It puts 8 output
columns per 256-thread CTA and reduces K cooperatively across all 256 threads, so
the grid is only `ceil(N/8) = 448` CTAs. The tuned kernel is register-limited to
~6 resident CTAs/SM (sm_90), i.e. ~792 CTAs = one wave, so 448 CTAs leaves ~43%
of the SM block slots idle on this **latency-bound** (Long-Scoreboard /
dependent-global-load) M=1 GEMV.

Every output column is reduced **entirely within one CTA** by all 256 threads
striding the same K tiles in the same order. So reducing the columns-per-CTA is a
pure grid-fill knob: the fp32 accumulation order per column is unchanged and the
output is **bit-identical**; only the CTA count changes (`grid = ceil(N/COLS)`).
Fewer columns/CTA ⇒ more CTAs ⇒ more resident warps across more SMs ⇒ the
dependent-load latency is hidden.

Implemented as `matmul_nbits_gemv_f16_scales_f16_down_tpl<COLS>` with extern-C
instantiations for `COLS ∈ {8, 4, 2}` (the byte-exact `_down`, `_down_c4`,
`_down_c2`). The host picks `COLS` from the device multiprocessor count
(`select_down_columns`): keep the largest `COLS` whose grid already meets a
~2-wave per-SM CTA target (`SM_count × 12`), floored at 2. Wide down projections
keep the cheaper 8-column launch; narrow (grid-starved) ones split. No per-model
magic — keys only on `N` and the SM count — and the choice is a launch-time
constant, safe to record into / replay from a CUDA graph.

## A/B sweep (columns per CTA)

Median of 3 alternating trials, H200 idle GPU, `CUDA_VISIBLE_DEVICES=<idle>
taskset -c 1`, `--steady --warmups 2 --runs 3 --tokens 128`. `ONNX_GENAI_DOWN_COLS`
forces the width; unset = the deployed `select_down_columns` (which chooses 2
here).

| Cols/CTA | Grid | Waves/SM | tok/s (median) | vs 8 |
|---:|---:|---:|---:|---:|
| 8 (baseline) | 448 | 0.57 | 301.76 | — |
| 4 | 896 | 1.13 | 305.76 | +1.3% |
| **2 (deployed)** | **1792** | **2.26** | **308.04** | **+2.08%** |
| 1 | 3584 | 4.5 | ~302 | ~0% (regressed vs 2) |

`COLS=1` erases the gain: the 256-thread CTA reduces a single column, so the
activation is re-read 8× and the grid over-subscribes — occupancy up but wasted
traffic cancels it. `COLS=2` (~2 waves) is the sweet spot, hence the floor.

Raw deployed A/B (baseline vs AUTO), GPU 3:

```
trial1 baseline=301.76 AUTO=308.13
trial2 baseline=302.20 AUTO=308.04
trial3 baseline=301.69 AUTO=307.91
```

Reproduced on GPUs 3, 5, 6 (+2.0–2.2% each); the only off-trend samples were
transient host-contention outliers (e.g. a lone 261 tok/s), excluded per method.
Intra-group spread of the clean trials is < 0.5 tok/s.

## ncu evidence (down kernel, `--graph-profiling node`)

| Metric | 8 cols (before) | 2 cols (after) |
|---|---:|---:|
| grid size | 448 | 1792 |
| waves/SM | 0.57 | **2.26** |
| SM throughput | 35.7% | **42.5%** |
| compute-mem throughput | 32.7% | **47.0%** |
| warps active | 39.5% | **62.5%** |

Occupancy, memory throughput, and warp residency all rise — the idle SMs are now
doing work and the latency-bound GEMV hides more of its dependent-load latency.

## Correctness

- Greedy decode token IDs (128 tokens, Qwen2.5-7B) are **byte-identical** to the
  8-column baseline — the change is bit-exact by construction (per-column
  reduction order unchanged).
- `fp16_down_projection_is_bit_exact_to_staged_kernel` extended to a
  `_c4`-selecting shape (K=16384, N=8192); it already exercised the `_c2` path.
  All 19 `matmul_nbits` lib tests pass, plus a new pure `select_down_columns`
  heuristic test.
- Pre-existing, unrelated: two `int8_block32` GPU tolerance tests fail on the
  clean base commit too (an int8 rounding tolerance issue on this toolchain);
  not touched by this change.

## Host / method

- Source branch: `perf/qwen7b-decode-hotkernel` off `origin/main` `1160f321`.
- Model: `~/.foundry/cache/models/Microsoft/qwen2.5-7b-instruct-cuda-gpu-4/v4`
  (genuine CUDA-EP int4, fp16 activation/scales).
- Pinning: idle H200 (not GPU 1, which held the other team's ~129.6 GB), one
  physical GPU via `CUDA_VISIBLE_DEVICES`, CPU 1 via `taskset -c 1`.

## Exact commands

```bash
source /home/justinchu/onnx-genai/.cudaenv.sh
cd /home/justinchu/wt-drake-perf
cargo build --release -p onnx-genai-bench --features bench-native,bench-ort,cuda --bin profile_native

QWEN7=~/.foundry/cache/models/Microsoft/qwen2.5-7b-instruct-cuda-gpu-4/v4
BIN=./target/release/profile_native
COMMON="--ep cuda --steady --warmups 2 --runs 3 --tokens 128"

# A/B (force width with ONNX_GENAI_DOWN_COLS=8|4|2; unset = deployed auto)
for t in 1 2 3; do
  CUDA_VISIBLE_DEVICES=<idle> taskset -c 1 env ONNX_GENAI_DOWN_COLS=8 $BIN --model "$QWEN7" $COMMON
  CUDA_VISIBLE_DEVICES=<idle> taskset -c 1                            $BIN --model "$QWEN7" $COMMON
done
```
