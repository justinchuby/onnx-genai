# Native CPU EP vs. onnxruntime-genai CPU — 2026-07-25

## Headline

The reconciliation changes the initial small-model-only story, but it does not
produce a native win under the requested matched pinning. With the persistent
pool unset, the auto-calibrator behaves like the flat path and native reaches
**0.36× ORT on Qwen3 0.6B** and **0.27× on Qwen3.5 2B**. Forcing 16 workers into
an exactly 16-CPU cpuset, or 32 workers into an exactly 32-CPU cpuset, starves
the inline dispatcher and collapses native to **0.01–0.05× ORT**.

The requested 7B/32-thread control therefore does **not** reproduce the prior
win: it is **1.47 vs. 30.77 tok/s (0.048×)** with `taskset -c 0-31`. An
additional diagnostic that keeps 32 workers but allows CPUs `0-47` restores
native to **29.48 vs. 30.59 tok/s (0.964×)**. The dominant reconciliation
finding is cpuset headroom, not model size alone.

## Method

- Current branch: `73b3c6e4f4641bd89a9b50fc558a1f3240eb1ebc`.
- Runtime base: current `origin/main`
  `b33ebc88bcbdcd2d83798f29461ff23a2b69e1b1`; the branch changes only the
  benchmark harness and this report.
- Real Foundry `generic-cpu` int4 ONNX artifacts; raw 510-token prompt.
- Greedy decoding, exactly 128 new tokens, EOS suppressed until that length.
- One complete warmup generation per process, then one measured generation.
- Three independent measured processes per cell; tables show median [min–max].
- Steady decode excludes the first eight emitted tokens.
- Every process launched only after 1-minute load average was below 10.
- Because the shared host could jump from single-digit to triple-digit load
  during a run, the reconciliation script also rejected a process if final
  load exceeded `2 × pinned_threads + 10`. One 2B forced-pool attempt was
  rejected and rerun; no sample was excluded based on throughput.
- The initially noisy 0.6B/16-thread and 7B/32-thread ORT cells were repeated
  from scratch while polling load every 100 ms. Their accepted peak 1-minute
  loads were 10.49 and 18.12 respectively; the resulting ranges are tight.

## Decode scoreboard

`auto` means `ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL` was unset. `off` means
explicit `=0`; `on` means forced `=1`.

| Model | Threads / allowed CPUs | Native pool | Native tok/s | ORT GenAI tok/s | Native/ORT |
|---|---|---|---:|---:|---:|
| Qwen3 0.6B | 16 / `0-15` | auto | 38.01 [37.21–38.18] | 105.01 [102.03–109.95] | **0.362×** |
| Qwen3 0.6B | 16 / `0-15` | off | 37.71 [33.32–38.21] | 105.01 [102.03–109.95] | **0.359×** |
| Qwen3 0.6B | 16 / `0-15` | on | 1.12 [1.12–1.13] | 105.01 [102.03–109.95] | **0.011×** |
| Qwen3 0.6B | 32 / `0-31` | on | 1.11 [1.11–1.11] | 102.12 [94.06–102.47] | **0.011×** |
| Qwen3.5 2B text | 16 / `0-15` | auto | 15.68 [15.54–16.01] | 58.70 [58.66–59.75] | **0.267×** |
| Qwen3.5 2B text | 16 / `0-15` | off | 15.72 [13.60–15.89] | 58.70 [58.66–59.75] | **0.268×** |
| Qwen3.5 2B text | 16 / `0-15` | on | 1.53 [1.53–1.54] | 58.70 [58.66–59.75] | **0.026×** |
| Qwen3.5 2B text | 32 / `0-31` | on | 1.41 [1.41–1.41] | 66.16 [65.51–67.24] | **0.021×** |
| Qwen2.5-Coder 7B | 32 / `0-31` | on | 1.47 [1.46–1.47] | 30.77 [30.50–30.90] | **0.048×** |
| Qwen2.5-Coder 7B | 32 / `0-47` | on | 29.48 [29.06–29.59] | 30.59 [30.43–30.82] | **0.964×** |

The auto and explicit-flat small-model medians differ by less than 1%, while
forced mode is stable at the collapsed rate. The corresponding logs say:

```text
auto: persistent SPMD decode pool built for auto-calibration ... the flat path stays committed under load
on:   persistent SPMD decode pool forced on ... (always dispatches to the pool)
```

That output plus the auto/off agreement confirms that the small-model auto runs
used the flat behavior. Forcing the pool did change the result, but negatively:
0.6B retained only **2.9%** of its auto throughput and 2B retained **9.8%**.

## 7B prefill control

Prefill is `510 / time-to-first-token` and includes first-token decode.

| Allowed CPUs | Native tok/s | ORT GenAI tok/s | Native/ORT |
|---|---:|---:|---:|
| `0-31` | 92.08 [86.51–92.33] | 179.90 [175.51–182.69] | **0.512×** |
| `0-47` headroom diagnostic | 92.56 [92.27–92.73] | 175.92 [174.78–176.08] | **0.526×** |

The prior approximately 25-vs-21 tok/s milestone is not reproduced by the exact
32-CPU control. Native itself reaches 29.5 tok/s when the dispatcher has
headroom, but ORT GenAI reaches 30.6 tok/s in the same run set, leaving native
near parity rather than ahead. This is not evidence that the int4 kernel fell
from 25 tok/s; it is evidence that full cpuset subscription is pathological and
that the matched ORT baseline is higher for this prompt/harness.

## Small-model per-op profile

One Qwen3 0.6B run used 16 workers, CPUs `0-15`, forced persistent mode, and
`ONNX_GENAI_PROFILE_OPS=1`. The table is the median of the 15 measured
post-prefill decode steps; each step executes 197 `MatMulNBits` and 28
`GroupQueryAttention` nodes.

| Op | Median ms/step | Share |
|---|---:|---:|
| `MatMulNBits` | 786.048 | **87.13%** |
| `GroupQueryAttention` | 112.393 | **12.47%** |
| `SimplifiedLayerNormalization` | 1.308 | 0.15% |
| `Reshape` | 0.860 | 0.10% |
| `SkipSimplifiedLayerNormalization` | 0.599 | 0.07% |
| All remaining ops | <0.3 each | 0.08% combined |

Median node execution was 902.637 ms/step [890.731–902.814], consistent with
the profiled 1.11 tok/s. The bucket is dominated by `MatMulNBits`, but the
32-worker/48-CPU 7B recovery proves this is not simply slow AVX-512 VNNI math:
the forced pool waits inside each `MatMulNBits` dispatch when workers consume
every allowed CPU and starve the inline dispatcher. The first optimization
target is therefore cpuset-aware worker sizing that reserves dispatcher
headroom; `GroupQueryAttention` is the clear second target.

## Token exactness

Every native mode and every ORT comparison produced the same complete 128-token
array in all paired runs. This includes auto, explicit flat, forced persistent,
16/32 threads, the requested 7B control, and the 48-CPU headroom diagnostic.
The first 16 IDs were:

```text
Qwen3 0.6B / Qwen2.5-Coder 7B:
576, 3974, 13876, 38835, 34208, 916, 279, 15678,
5562, 13, 576, 3974, 13876, 38835, 34208, 916

Qwen3.5 2B:
561, 3841, 13477, 37550, 33075, 888, 279, 15217,
5388, 13, 561, 3841, 13477, 37550, 33075, 888
```

## Host and runtime

- Dual-socket Intel Xeon Platinum 8480C: 96 cores, one thread/core, two NUMA
  nodes, 210 MiB aggregate L3.
- ISA: AVX2, AVX-VNNI, AVX-512 VNNI, AVX-512 BF16/FP16, AMX INT8/BF16.
- Native int4 decode dispatch: **`DotKernel::Avx512Vnni`**.
- Native build: `mlas`; ORT comparison: `onnxruntime-genai 0.14.1`,
  `onnxruntime 1.26.0`, CPU provider.
- Native thread controls: `RAYON_NUM_THREADS=N`,
  `ONNX_GENAI_CPU_DECODE_THREADS=N`; ORT: intra-op `N`, inter-op 1.

| Model | Foundry artifact | ONNX payload |
|---|---|---:|
| Qwen3 0.6B | `Microsoft/qwen3-0.6b-generic-cpu-4/v4` | 524,589,133 bytes |
| Qwen3.5 2B text | `Microsoft/qwen3.5-2b-text-generic-cpu-1/v1` | 1,515,833,911 bytes |
| Qwen2.5-Coder 7B | `Microsoft/qwen2.5-coder-7b-instruct-generic-cpu-4/v4` | 6,617,226,822 bytes |

## Reproduce

```bash
cd /home/justinchu/onnx-genai-cpu-bench
cargo build --release -p onnx-genai-bench \
  --features mlas --bin profile_native

export LD_LIBRARY_PATH="$(
  find target -type d \
    -path '*onnx-genai-ort-sys*/out/ort-prebuilt/lib' | head -1
):${LD_LIBRARY_PATH:-}"

PROMPT=$(python3 - <<'PY'
print(('The quick brown fox jumps over the lazy dog. ' * 51).strip())
PY
)

guard_load() {
  while awk -v value="$(cut -d' ' -f1 /proc/loadavg)" \
    'BEGIN { exit !(value >= 10.0) }'
  do
    sleep 30
  done
  cat /proc/loadavg
}

run_native() {
  local model=$1 threads=$2 cpus=$3 pool=$4
  guard_load
  ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL="$pool" \
  RAYON_NUM_THREADS="$threads" \
  ONNX_GENAI_CPU_DECODE_THREADS="$threads" \
    taskset -c "$cpus" ./target/release/profile_native \
      --model "$model" --ep cpu --backend native --steady \
      --warmups 1 --runs 1 --tokens 128 --decode-skip 8 \
      --prompt "$PROMPT"
}

run_native_auto() {
  local model=$1
  guard_load
  env -u ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL \
    RAYON_NUM_THREADS=16 ONNX_GENAI_CPU_DECODE_THREADS=16 \
    taskset -c 0-15 ./target/release/profile_native \
      --model "$model" --ep cpu --backend native --steady \
      --warmups 1 --runs 1 --tokens 128 --decode-skip 8 \
      --prompt "$PROMPT"
}

run_ort() {
  local model=$1 threads=$2 cpus=$3
  guard_load
  OGA_RAW=1 OGA_WARMUPS=1 OGA_RUNS=1 \
  OGA_THREADS="$threads" OGA_DECODE_SKIP=8 \
    taskset -c "$cpus" python3 examples/benchmark/oga_bench.py \
      "$model" "$PROMPT" 128
}

QWEN06=/home/justinchu/.foundry/cache/models/Microsoft/qwen3-0.6b-generic-cpu-4/v4
QWEN2=/home/justinchu/.foundry/cache/models/Microsoft/qwen3.5-2b-text-generic-cpu-1/v1
QWEN7=/home/justinchu/.foundry/cache/models/Microsoft/qwen2.5-coder-7b-instruct-generic-cpu-4/v4

# Repeat every cell three times, alternating order.
run_native_auto "$QWEN06"
run_native "$QWEN06" 16 0-15 0
run_native "$QWEN06" 16 0-15 1
run_native "$QWEN06" 32 0-31 1
run_ort "$QWEN06" 16 0-15
run_ort "$QWEN06" 32 0-31

run_native_auto "$QWEN2"
run_native "$QWEN2" 16 0-15 0
run_native "$QWEN2" 16 0-15 1
run_native "$QWEN2" 32 0-31 1
run_ort "$QWEN2" 16 0-15
run_ort "$QWEN2" 32 0-31

run_native "$QWEN7" 32 0-31 1
run_ort "$QWEN7" 32 0-31

# Diagnostic that reserves 16 allowed CPUs beyond the 32 pool workers.
run_native "$QWEN7" 32 0-47 1
run_ort "$QWEN7" 32 0-47

# Per-op diagnostic; aggregate the last 15 post-prefill tables.
guard_load
ONNX_GENAI_PROFILE_OPS=1 \
ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL=1 \
RAYON_NUM_THREADS=16 ONNX_GENAI_CPU_DECODE_THREADS=16 \
  taskset -c 0-15 ./target/release/profile_native \
    --model "$QWEN06" --ep cpu --backend native --steady \
    --warmups 1 --runs 1 --tokens 16 --decode-skip 8 \
    --prompt "$PROMPT"
```

## Verdict

Native still loses on the 0.6B and 2B models when auto selects the flat path,
and adding cores does not help if a forced persistent pool occupies every
allowed CPU. The exact requested 7B control also loses catastrophically for
that reason, while reserving dispatcher headroom moves the same native build to
**0.964× ORT**, near parity but not the prior win. The profile assigns 87% of
the collapsed small-model step to `MatMulNBits` and 12.5% to GQA, but the
headroom A/B identifies persistent-dispatch starvation as the first fix;
attention is the second optimization target.
