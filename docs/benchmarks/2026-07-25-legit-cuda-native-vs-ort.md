# Legitimate CUDA native vs. ORT sweep — 2026-07-25

## Headline

All three requested models used Foundry `cuda-gpu` artifacts. Their ORT runs
showed substantial device allocation and sustained H200 utilization, and none
emitted the `Memcpy nodes are added` warning that invalidated the earlier
`generic-cpu` Phi/Coder comparison. These are valid native-CUDA-vs-ORT-CUDA
measurements for the exact artifacts:

- Phi-4-mini: native **322.04 tok/s** vs. ORT **232.58 tok/s**,
  **1.385× (+38.5%)**.
- Qwen2.5-1.5B: native **632.62 tok/s** vs. ORT **435.66 tok/s**,
  **1.452× (+45.2%)**.
- Qwen2.5-7B: native **302.24 tok/s** vs. ORT **274.75 tok/s**,
  **1.100× (+10.0%)**.

The 7B result is a modest 10.0% win, not a multi-x claim.

## Measurements

Values are medians of three runs. Parentheses contain the min–max tok/s range
and full-range span as a percentage of the median. Every process used two
warmups, generated 128 tokens, and excluded the first eight emitted tokens from
the steady-decode interval.

| Model | Native `model` | Native `fp16` | ORT CUDA | model/ORT | fp16/ORT | fp16/model |
|---|---:|---:|---:|---:|---:|---:|
| Phi-4-mini | 322.04 (321.33–322.08; 0.23%) | 322.33 (322.22–322.41; 0.06%) | 232.58 (232.07–234.63; 1.10%) | **1.385×** | **1.386×** | 1.001× |
| Qwen2.5-1.5B | 632.62 (632.26–632.86; 0.09%) | 632.68 (632.57–633.15; 0.09%) | 435.66 (432.03–436.09; 0.93%) | **1.452×** | **1.452×** | 1.000× |
| Qwen2.5-7B | 302.24 (301.91–302.32; 0.14%) | 302.25 (301.83–302.25; 0.14%) | 274.75 (273.40–275.99; 0.94%) | **1.100×** | **1.100×** | 1.000× |

All three CUDA artifacts already use fp16 activation/scales. The opt-in rewrite
only matches fp32-scale `MatMulNBits`, so `--decode-precision fp16` was the
documented no-op here. Native `model` and `fp16` produced identical token IDs;
their 0.0–0.1% median differences are measurement noise.

## ORT-on-GPU validity

Physical H200 GPU 6 was sampled every 100 ms throughout each ORT process:

| Model | Peak GPU utilization | Peak GPU memory | Nonzero-utilization samples | `Memcpy` warning | Verdict |
|---|---:|---:|---:|---|---|
| Phi-4-mini | 88% | 5,289 MiB | 48 | None | **Valid GPU run** |
| Qwen2.5-1.5B | 86% | 2,727 MiB | 26 | None | **Valid GPU run** |
| Qwen2.5-7B | 91% | 5,797 MiB | 46 | None | **Valid GPU run** |

ORT printed its standard `VerifyEachNodeIsAssignedToAnEp` message explaining
that shape-related operators may be intentionally assigned to CPU. That message
appeared without any inserted-`Memcpy` warning, while the monitor recorded
86–91% GPU utilization during decode. This is not the prior fallback-thrash
pattern: the model computation demonstrably ran on GPU.

## Host state

- Source: `5a8c3dc9fb3be563056d490f97ca27bb69cdfec4`.
- Pinning: physical GPU 6 via `CUDA_VISIBLE_DEVICES=6`, CPU 1 via
  `taskset -c 1`.
- GPU 6 was 0 MiB allocated and 0% utilized before every configuration. It
  returned to 0 MiB after every process. GPU 1 retained the other team's
  approximately 129.6 GB allocation and was never selected.
- Final host load average was 4.52 on 96 logical CPUs; a three-second sample
  found pinned CPU 1 100% idle after the sweep. The tight within-configuration
  spreads are consistent with an uncontended run.

## Generated-text sanity

No model failed to load. Each backend was deterministic across its three
measured runs.

- Phi-4-mini produced the same coherent Dockerfile/Gunicorn continuation in all
  three configurations.
- Qwen2.5-7B produced the same coherent C string-reversal program in all three
  configurations.
- **Qwen2.5-1.5B was not fully coherent:** native and ORT both began with
  readable prose about popular books, then repeatedly emitted “The list is
  sorted by the number of copies sold.” Native `model` and `fp16` were
  token-identical; ORT differed in an early source attribution but showed the
  same repetition failure. The throughput row remains a valid GPU-vs-GPU
  measurement, but this output must not be presented as a clean coherence
  result.

The existing GPU-gated correctness locks remain the correctness authority; this
was only the requested output eyeball.

## Exact commands

```bash
source /home/justinchu/onnx-genai/.cudaenv.sh
cd /home/justinchu/wt-ripley-legit
cargo build --release -p onnx-genai-bench \
  --features bench-native,bench-ort,cuda --bin profile_native

BIN=./target/release/profile_native
COMMON="--ep cuda --steady --warmups 2 --runs 3 --tokens 128"

PHI4=/home/justinchu/.foundry/cache/models/Microsoft/Phi-4-mini-instruct-cuda-gpu-5/v5
QWEN15=/home/justinchu/.foundry/cache/models/Microsoft/qwen2.5-1.5b-instruct-cuda-gpu-4/v4
QWEN7=/home/justinchu/.foundry/cache/models/Microsoft/qwen2.5-7b-instruct-cuda-gpu-4/v4

for MODEL in "$PHI4" "$QWEN15" "$QWEN7"; do
  CUDA_VISIBLE_DEVICES=6 taskset -c 1 $BIN --model "$MODEL" $COMMON \
    --backend native --decode-precision model
  CUDA_VISIBLE_DEVICES=6 taskset -c 1 $BIN --model "$MODEL" $COMMON \
    --backend native --decode-precision fp16
  CUDA_VISIBLE_DEVICES=6 taskset -c 1 $BIN --model "$MODEL" $COMMON \
    --backend ort --decode-precision model
done
```

During each ORT command, physical GPU 6 was independently monitored with:

```bash
nvidia-smi -i 6 \
  --query-gpu=timestamp,memory.used,utilization.gpu,utilization.memory \
  --format=csv,noheader,nounits -lms 100
```
