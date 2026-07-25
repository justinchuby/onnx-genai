# Native CPU EP vs. onnxruntime-genai CPU — 2026-07-25

## Headline

On these real Foundry `generic-cpu` int4 models, native is slower than
onnxruntime-genai in every measured cell. Native reaches **0.18–0.20× ORT
prefill throughput** and **0.24–0.44× ORT steady-decode throughput**. The
generated token IDs are exact matches; this is a performance gap, not a
different-sequence comparison.

## Measurements

Values are medians of three independent processes; brackets contain the full
min–max range. Each process discarded one complete warmup generation. The raw
prompt is 510 tokens in both tokenizers. Prefill throughput is
`510 / time-to-first-token`; steady decode excludes the first eight emitted
tokens. No measured sample was discarded.

| Model | New tokens | Phase | Native tok/s | ORT GenAI tok/s | Native/ORT |
|---|---:|---|---:|---:|---:|
| Qwen3 0.6B int4 | 128 | Prefill | 292.00 [258.40–295.70] | 1,456.98 [820.60–1,461.49] | **0.200×** |
| Qwen3 0.6B int4 | 128 | Decode | 38.17 [29.76–38.69] | 106.38 [67.38–110.41] | **0.359×** |
| Qwen3 0.6B int4 | 512 | Prefill | 293.53 [256.98–296.11] | 1,448.84 [1,442.51–1,457.61] | **0.203×** |
| Qwen3 0.6B int4 | 512 | Decode | 24.38 [21.54–27.78] | 101.06 [100.00–101.36] | **0.241×** |
| Qwen3.5 2B text int4 | 128 | Prefill | 53.70 [53.45–57.72] | 282.15 [263.32–309.16] | **0.190×** |
| Qwen3.5 2B text int4 | 128 | Decode | 13.58 [13.52–13.71] | 31.18 [31.06–36.54] | **0.436×** |
| Qwen3.5 2B text int4 | 512 | Prefill | 58.69 [56.28–59.92] | 319.07 [291.60–373.46] | **0.184×** |
| Qwen3.5 2B text int4 | 512 | Decode | 13.95 [13.36–14.07] | 39.94 [36.94–59.00] | **0.349×** |

The wider ranges are reported rather than hidden: every process began below
the required load threshold, but the shared host occasionally became busy
during a run. The medians and the large gap agree across both token lengths and
models, so contention does not change the verdict.

## Correctness

Both runtimes used greedy decoding, EOS stopping was disabled/blocked until the
requested length, and the raw prompt was identical. Native and ORT produced the
same complete 128- and 512-token ID arrays in all paired runs. The first 16 IDs
were:

- Qwen3 0.6B: `576, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562, 13, 576, 3974, 13876, 38835, 34208, 916`
- Qwen3.5 2B: `561, 3841, 13477, 37550, 33075, 888, 279, 15217, 5388, 13, 561, 3841, 13477, 37550, 33075, 888`

`oga_bench.py` now prints the generated IDs and per-run phase timing, and fails
if measured greedy runs differ.

## Host and runtime

- Source: `26df3f52e6ce7b414449552bb1d8117a6ab5b6d9`.
- CPU: dual-socket Intel Xeon Platinum 8480C, 96 CPUs, 48 cores/socket,
  one thread/core, 210 MiB aggregate L3, two NUMA nodes.
- ISA: AVX2, AVX-VNNI, AVX-512 VNNI, AVX-512 BF16/FP16, AMX INT8/BF16.
- Native int4 decode dispatch: **`DotKernel::Avx512Vnni`**. The runtime selects
  it when AVX2, AVX512F/BW/VL, and AVX512VNNI are present; all were detected.
- Threads: both runtimes were restricted to CPUs `0-15`. Native used
  `RAYON_NUM_THREADS=16` and `ONNX_GENAI_CPU_DECODE_THREADS=16`; ORT used
  intra-op 16 and inter-op 1.
- Native build: `mlas` feature enabled. ORT comparison:
  `onnxruntime-genai 0.14.1`, `onnxruntime 1.26.0`, CPU provider.
- Every measured process was launched only after 1-minute load average fell
  below 10; accepted launch values ranged from 4.32 to 9.82.

## Models

| Model | Foundry artifact | ONNX payload |
|---|---|---:|
| Qwen3 0.6B | `Microsoft/qwen3-0.6b-generic-cpu-4/v4` | 524,589,133 bytes |
| Qwen3.5 2B text | `Microsoft/qwen3.5-2b-text-generic-cpu-1/v1` | 1,515,833,911 bytes |

The 7B artifact was not run: the shared host repeatedly spiked well above the
load guard, and the 2B native sweep already required long clean windows. The
requested 0.6B + 2B minimum is complete.

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
    cat /proc/loadavg
    sleep 30
  done
  cat /proc/loadavg
}

run_native() {
  local model=$1 tokens=$2
  guard_load
  RAYON_NUM_THREADS=16 ONNX_GENAI_CPU_DECODE_THREADS=16 \
    taskset -c 0-15 ./target/release/profile_native \
      --model "$model" --ep cpu --backend native --steady \
      --warmups 1 --runs 1 --tokens "$tokens" --decode-skip 8 \
      --prompt "$PROMPT"
}

run_ort_genai() {
  local model=$1 tokens=$2
  guard_load
  OGA_RAW=1 OGA_WARMUPS=1 OGA_RUNS=1 \
    OGA_THREADS=16 OGA_DECODE_SKIP=8 \
    taskset -c 0-15 python3 examples/benchmark/oga_bench.py \
      "$model" "$PROMPT" "$tokens"
}

QWEN06=/home/justinchu/.foundry/cache/models/Microsoft/qwen3-0.6b-generic-cpu-4/v4
QWEN2=/home/justinchu/.foundry/cache/models/Microsoft/qwen3.5-2b-text-generic-cpu-1/v1

for model in "$QWEN06" "$QWEN2"; do
  for tokens in 128 512; do
    for repetition in 1 2 3; do
      if (( repetition % 2 == 1 )); then
        run_native "$model" "$tokens"
        run_ort_genai "$model" "$tokens"
      else
        run_ort_genai "$model" "$tokens"
        run_native "$model" "$tokens"
      fi
    done
  done
done
```

## Verdict

Native loses clearly here: ORT is about **5× faster in prefill** and
**2.3–4.1× faster in steady decode**. Native's best relative result is the 2B
128-token decode at **0.436× ORT**; there are no ties or native wins in this
CPU sweep.
