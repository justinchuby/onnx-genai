# Qwen2.5-1.5B and Qwen2.5-7B int4 H200 decode — 2026-07-22

## Results

Both cached Foundry packages loaded and completed native CUDA decode. Results are
the median of three measured generations after two discarded warmups.
`--steady --decode-skip 8` excludes prefill, graph capture, and the first eight
emitted tokens.

| Model | Output tokens | Per-run tok/s | Median tok/s | Median session-step latency | HBM roofline | Achieved |
|---|---:|---|---:|---:|---:|---:|
| Qwen2.5-1.5B int4 | 128 | 487.66, 488.46, 484.95 | **487.66** | **2.051 ms/token** | 3,811 tok/s | **12.8%** |
| Qwen2.5-1.5B int4 | 1024 | 458.36, 457.88, 457.76 | **457.88** | **2.184 ms/token** | 3,756 tok/s | **12.2%** |
| Qwen2.5-7B int4 | 128 | 230.25, 230.99, 230.47 | **230.47** | **4.339 ms/token** | 840 tok/s | **27.5%** |
| Qwen2.5-7B int4 | 1024 | 223.53, 223.38, 223.36 | **223.38** | **4.477 ms/token** | 834 tok/s | **26.8%** |

`profile_native` calls this field `decode` rather than `ort.session_run`; it is
the native backend's available steady session-step latency over the same window
as the reported throughput.

## CUDA configuration and packages

- Source: `origin/main` at `3bc4ef0c3628663847ebf34bbf5054189c9bddac`.
- GPU: NVIDIA H200, 143,771 MiB, driver 580.105.08; physical GPU 0.
- Qwen2.5-1.5B:
  `/home/justinchu/.foundry/cache/models/Microsoft/qwen2.5-1.5b-instruct-cuda-gpu-4/v4`
- Qwen2.5-7B:
  `/home/justinchu/.foundry/cache/models/Microsoft/qwen2.5-7b-instruct-cuda-gpu-4/v4`
- Prompt: `The capital of France is` (five tokens).
- Greedy decode, temperature 0, EOS stopping disabled.
- `ONNX_GENAI_DEVICE_KV=1`, `ONNX_GENAI_CUDA_GRAPH=1`,
  `ONNX_GENAI_REQUIRE_CUDA=1`, and `CUDA_VISIBLE_DEVICES=0`.
- A separate 32-token diagnostic for each package reported
  `enabled=true`, one measured capture, 29 measured replays, zero fallbacks,
  and zero measured KV H2D/D2H calls or bytes. No eager fallback was needed.

The diagnostic continuations began `" Paris. The capital of France..."` for
1.5B and `" Paris. It is located..."` for 7B.

## Roofline arithmetic

The estimate uses the requested effective H200 bandwidth
`B = 3.35e12 bytes/s` and:

```text
roofline_tok/s = B / (streamed_weight_bytes + average_fp16_KV_read_bytes)
KV bytes per cached token =
    layers * 2(K,V) * kv_heads * head_dim * sizeof(fp16)
```

Streamed weight bytes are the exact ONNX initializer bytes excluding the full
fp16 token-embedding table, because decode reads only one embedding row. This
retains packed int4 matrices, their fp16 scales, normalization constants, and
the quantized LM head. The steady windows average 72.5 cached tokens at output
length 128 and 520.5 at output length 1024.

### Qwen2.5-1.5B

```text
streamed weights = 1,343,683,584 - 466,747,392
                 = 876,936,192 bytes
KV/cached token  = 28 * 2 * 2 * 128 * 2 = 28,672 bytes

128:  total = 876,936,192 + 72.5 * 28,672
            = 879,014,912 bytes/token
      roofline = 3.35e12 / 879,014,912 = 3,811 tok/s
      achieved = 487.66 / 3,811 = 12.8%

1024: total = 876,936,192 + 520.5 * 28,672
            = 891,859,968 bytes/token
      roofline = 3.35e12 / 891,859,968 = 3,756 tok/s
      achieved = 457.88 / 3,756 = 12.2%
```

### Qwen2.5-7B

```text
streamed weights = 5,076,085,760 - 1,089,994,752
                 = 3,986,091,008 bytes
KV/cached token  = 28 * 2 * 4 * 128 * 2 = 57,344 bytes

128:  total = 3,986,091,008 + 72.5 * 57,344
            = 3,990,248,448 bytes/token
      roofline = 3.35e12 / 3,990,248,448 = 840 tok/s
      achieved = 230.47 / 840 = 27.5%

1024: total = 3,986,091,008 + 520.5 * 57,344
            = 4,015,938,560 bytes/token
      roofline = 3.35e12 / 4,015,938,560 = 834 tok/s
      achieved = 223.38 / 834 = 26.8%
```

The supplied Qwen2.5-0.5B `886 tok/s` ceiling is a practical gap-free ceiling,
not the literal result of this explicit weight-plus-KV byte model: at
3.35 TB/s, 886 tok/s implies 3.781 GB moved per token. Its reported
91.4%/87.9% values therefore use a different denominator from the physical HBM
percentages above.

## Exact commands

```bash
cd /home/justinchu/wt-joi-bench

cargo build --release -p onnx-genai-bench \
  --features bench-native,cuda --bin profile_native

export LD_LIBRARY_PATH=$(python3 -c "import os,glob;print(':'.join(glob.glob(os.path.expanduser('~')+'/.conda/envs/*/lib/python*/site-packages/nvidia/*/lib')))"):/home/justinchu/onnx-genai/.ort-cuda-1.27/onnxruntime-linux-x64-1.27.0/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
export ONNX_GENAI_DEVICE_KV=1
export ONNX_GENAI_CUDA_GRAPH=1
export ONNX_GENAI_REQUIRE_CUDA=1

./target/release/profile_native \
  --model "$MODEL" --tokens "$TOKENS" --warmups 2 --runs 3 \
  --steady --decode-skip 8 --ep cuda \
  --prompt "The capital of France is"
```

The command was run once for each package path above with `TOKENS=128` and
`TOKENS=1024`, never concurrently. CUDA-graph/device-KV diagnostics used:

```bash
./target/release/profile_native \
  --model "$MODEL" --tokens 32 --warmups 1 --runs 1 --ep cuda \
  --prompt "The capital of France is"
```

## Baseline comparison

Qwen2.5-1.5B reached **487.66/457.88 tok/s** and Qwen2.5-7B reached
**230.47/223.38 tok/s** at 128/1024 tokens, below Qwen2.5-0.5B's
**810.06/778.59 tok/s** but above Phi-4-mini's **94.5/93.2 tok/s**; the larger
models sustain a higher fraction of the explicit HBM byte roofline.
