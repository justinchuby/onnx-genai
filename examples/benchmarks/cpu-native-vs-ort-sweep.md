# Native CPU EP vs. onnxruntime-genai CPU — corrected post-#154 results

## Headline

These measurements supersede the pre-#154 scoreboard. The earlier
**1.1–1.5 tok/s** forced-pool results were dispatcher starvation, not useful
kernel-performance measurements. Commit
`a6848d4` ([#154](https://github.com/justinchuby/onnx-genai/pull/154))
reserves dispatcher headroom inside the process cpuset. On the same exact
32-CPU cpuset that previously collapsed, Qwen2.5-Coder 7B now reaches
**29.1 tok/s forced** and **29.9 tok/s auto**, versus **30.5 tok/s for ORT
GenAI**: **0.95× and 0.98× ORT**, respectively.

The fix also restores the small-model forced pool, but it does not make native
faster than ORT: Qwen3 0.6B is about **0.63× ORT**, while Qwen3.5 2B is about
**0.33× ORT**. Auto may still commit the flat path after calibration on these
models; the table reports the real out-of-box behavior rather than assuming a
pool choice.

The ORT harness now explicitly overrides the model configuration with pure
greedy options, including `repetition_penalty=1.0`. Previously, the 7B model's
inherited `1.1` penalty made the policies asymmetric. All native and ORT runs
below produced the same complete 128 generated token IDs; the ORT harness was
given the native IDs and failed the process on any divergence.

## Method

- Runtime base: `origin/main` at `a6848d4` (the #154 headroom fix).
- Real Foundry `generic-cpu` int4 artifacts; bare 510-token prompt.
- Exactly 128 new tokens, EOS suppressed until that length.
- Pure greedy: sampling off, repetition penalty 1.0, temperature 1.0, top-k 0,
  and top-p 1.0.
- One discarded complete warmup and one measured generation per process.
- Three independent processes per cell; median [minimum–maximum] is reported.
- Steady decode excludes the first eight emitted tokens.
- Every process waited for 1-minute load average below 10. Runs were monitored
  and rejected if final load reached `2 × pinned_threads + 10`; no accepted
  sample was removed based on throughput.
- Native and ORT used identical `taskset` CPU sets and thread counts. ORT used
  intra-op N and inter-op 1.
- `auto` leaves `ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL` unset; `forced` sets it
  to `1`.

## Decode scoreboard

| Model | Threads / CPUs | Native mode | Native tok/s | ORT tok/s | Native/ORT |
|---|---|---|---:|---:|---:|
| Qwen3 0.6B | 16 / `0-15` | auto | 68.36 [68.00–68.78] | 109.08 [108.72–109.16] | **0.627×** |
| Qwen3 0.6B | 16 / `0-15` | forced | 68.64 [68.10–68.65] | 109.08 [108.72–109.16] | **0.629×** |
| Qwen3.5 2B text | 16 / `0-15` | auto | 19.64 [19.61–19.70] | 59.57 [59.21–60.15] | **0.330×** |
| Qwen3.5 2B text | 16 / `0-15` | forced | 19.57 [19.47–19.63] | 59.57 [59.21–60.15] | **0.329×** |
| Qwen2.5-Coder 7B | 32 / `0-31` | auto | 29.94 [29.74–30.05] | 30.52 [30.35–30.55] | **0.981×** |
| Qwen2.5-Coder 7B | 32 / `0-31` | forced | 29.10 [29.09–29.29] | 30.52 [30.35–30.55] | **0.953×** |

The important 7B control is the forced row: with 32 configured decode threads
and only CPUs `0-31` allowed, native remains near ORT rather than collapsing to
the old **0.048×** result. The measured median is just below 30 tok/s, so these
data support “near parity,” not a claimed native win.

## 7B prefill control

Prefill throughput is `510 / time-to-first-token` and includes first-token
decode.

| Mode | Native tok/s | ORT tok/s | Native/ORT |
|---|---:|---:|---:|
| auto | 91.70 [91.15–92.12] | 179.01 [178.35–179.28] | **0.512×** |
| forced | 91.19 [90.68–91.70] | 179.01 [178.35–179.28] | **0.509×** |

## Token exactness

Each model had exactly one generated-token array across all nine paired
measurements (three auto, three forced, and three ORT). The first 16 IDs were:

```text
Qwen3 0.6B / Qwen2.5-Coder 7B:
576, 3974, 13876, 38835, 34208, 916, 279, 15678,
5562, 13, 576, 3974, 13876, 38835, 34208, 916

Qwen3.5 2B:
561, 3841, 13477, 37550, 33075, 888, 279, 15217,
5388, 13, 561, 3841, 13477, 37550, 33075, 888
```

## Host and artifacts

- Dual-socket Intel Xeon Platinum 8480C: 96 cores, one thread/core, two NUMA
  nodes, 210 MiB aggregate L3.
- ISA: AVX2, AVX-VNNI, AVX-512 VNNI/BF16/FP16, AMX INT8/BF16.
- Native int4 decode dispatch: `DotKernel::Avx512Vnni`.
- Native build: `mlas`; ORT comparison: `onnxruntime-genai 0.14.1`,
  `onnxruntime 1.26.0`, CPU provider.

| Model | Foundry artifact |
|---|---|
| Qwen3 0.6B | `Microsoft/qwen3-0.6b-generic-cpu-4/v4` |
| Qwen3.5 2B text | `Microsoft/qwen3.5-2b-text-generic-cpu-1/v1` |
| Qwen2.5-Coder 7B | `Microsoft/qwen2.5-coder-7b-instruct-generic-cpu-4/v4` |

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
}

run_native() {
  local model=$1 threads=$2 cpus=$3 mode=$4 log=$5
  guard_load
  if [[ "$mode" == auto ]]; then
    env -u ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL \
      RAYON_NUM_THREADS="$threads" \
      ONNX_GENAI_CPU_DECODE_THREADS="$threads" \
      taskset -c "$cpus" ./target/release/profile_native \
        --model "$model" --ep cpu --backend native --steady \
        --warmups 1 --runs 1 --tokens 128 --decode-skip 8 \
        --prompt "$PROMPT" | tee "$log"
  else
    ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL=1 \
    RAYON_NUM_THREADS="$threads" \
    ONNX_GENAI_CPU_DECODE_THREADS="$threads" \
      taskset -c "$cpus" ./target/release/profile_native \
        --model "$model" --ep cpu --backend native --steady \
        --warmups 1 --runs 1 --tokens 128 --decode-skip 8 \
        --prompt "$PROMPT" | tee "$log"
  fi
}

run_ort() {
  local model=$1 threads=$2 cpus=$3 native_log=$4
  local ids
  ids=$(sed -n 's/^generated_token_ids: //p' "$native_log" | tail -1)
  guard_load
  OGA_RAW=1 OGA_WARMUPS=1 OGA_RUNS=1 \
  OGA_THREADS="$threads" OGA_DECODE_SKIP=8 \
  OGA_EXPECTED_TOKEN_IDS="$ids" \
    taskset -c "$cpus" python3 examples/benchmark/oga_bench.py \
      "$model" "$PROMPT" 128
}

QWEN06=/home/justinchu/.foundry/cache/models/Microsoft/qwen3-0.6b-generic-cpu-4/v4
QWEN2=/home/justinchu/.foundry/cache/models/Microsoft/qwen3.5-2b-text-generic-cpu-1/v1
QWEN7=/home/justinchu/.foundry/cache/models/Microsoft/qwen2.5-coder-7b-instruct-generic-cpu-4/v4

# Run each cell three times, alternating order. Monitor /proc/loadavg during
# each process and rerun if final load is >= 2 * threads + 10.
run_native "$QWEN06" 16 0-15 auto qwen06-native.log
run_native "$QWEN06" 16 0-15 forced qwen06-forced.log
run_ort "$QWEN06" 16 0-15 qwen06-native.log

run_native "$QWEN2" 16 0-15 auto qwen2-native.log
run_native "$QWEN2" 16 0-15 forced qwen2-forced.log
run_ort "$QWEN2" 16 0-15 qwen2-native.log

run_native "$QWEN7" 32 0-31 auto qwen7-native.log
run_native "$QWEN7" 32 0-31 forced qwen7-forced.log
run_ort "$QWEN7" 32 0-31 qwen7-native.log
```

## Verdict

The post-#154 result is materially different from the rejected pre-fix
scoreboard. Full-cpuset forced persistent decode is healthy again and 7B is
near ORT parity. Native still trails ORT on both small models and on 7B
prefill. The corrected greedy-policy and token-exactness gates make those
remaining gaps apples-to-apples.
