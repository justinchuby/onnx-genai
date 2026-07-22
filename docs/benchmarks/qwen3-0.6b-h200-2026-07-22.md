# Qwen3-0.6B int4 H200 decode — 2026-07-22

## Results

The exported package loaded on ONNX Runtime 1.27 CUDA and generated the coherent
20-token smoke continuation:

```text
 Paris, and the capital of Italy is Rome. The capital of France is also the capital of the
```

The requested eager decode configuration completed, but exposed a package/runtime
metadata gap that disables the shared-KV path. Results are aggregate throughput
over three measured generations after two discarded warmups.

| Package/configuration | Output tokens | Aggregate tok/s | Wall latency | `ort.bind_inputs` | Explicit HBM roofline | Achieved |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3 export as supplied | 128 | **197.49** | 5.064 ms/token | 1.093 ms/token | 12,585 tok/s | **1.57%** |
| Qwen3 export as supplied | 1024 | **64.01** | 15.622 ms/token | 6.938 ms/token | 10,549 tok/s | **0.61%** |
| Qwen3 metadata diagnostic | 128 | **429.95** | 2.326 ms/token | 0.054 ms/token | 12,585 tok/s | **3.42%** |
| Qwen3 metadata diagnostic | 1024 | **381.49** | 2.621 ms/token | 0.058 ms/token | 10,549 tok/s | **3.62%** |

The diagnostic changed only metadata: `attention.type` from `grouped_query` to
the runtime-recognized `grouped_query_attention`, and added
`kv_cache.native_dtype: float16`. It is not the as-exported result. It confirms
that the severe context-length collapse is primarily failure to select the
fixed-capacity shared-KV path, not a QK-norm correctness problem.

## Model, package, and CUDA configuration

- Source: `origin/main` at `8d9d2fa279a0463b9fc6c02932ebd7b9ec775fb4`.
- GPU: NVIDIA H200, 143,771 MiB, driver 580.105.08; `CUDA_VISIBLE_DEVICES=0`.
- Package:
  `/home/justinchu/mobius/.scratch/qwen3-0.6b-int4-cuda`.
- Present: `model.onnx`, `model.onnx.data`, `inference_metadata.yaml`, and
  `tokenizer.json`.
- Missing: `genai_config.json`, `tokenizer_config.json`, and a chat template.
  Native `inference_metadata.yaml` is readable and is the preferred onnx-genai
  metadata source, so the missing compatibility config did not prevent loading.
- Architecture: Qwen3, 28 layers, 16 query heads, 8 KV heads, head dimension
  128, maximum sequence length 40,960, and RoPE cache length 40,960.
- Graph: 543 nodes, including 196 symmetric int4 `MatMulNBits` nodes
  (`block_size=32`, `bits=4`, `accuracy_level=4`), 28
  `GroupQueryAttention` nodes, 57 `RMSNormalization` nodes, and 56 explicit
  per-layer Q/K RMSNorm nodes.
- Prompt: raw `The capital of France is` (five tokenizer tokens), because the
  package has no chat template.
- Greedy decode, temperature 0, EOS stopping disabled.
- `ONNX_GENAI_DEVICE_KV=1`, `ONNX_GENAI_CUDA_GRAPH=0`,
  `ONNX_GENAI_REQUIRE_CUDA=1`, and ONNX Runtime 1.27 CUDA EP.

The requested
`onnxruntime-linux-x64-1.27.0/lib` directory contains no
`libonnxruntime_providers_cuda.so`; the CUDA-enabled runtime is under
`.ort-cuda-1.27/root/lib`, which was added to `ORT_ROOT` and
`LD_LIBRARY_PATH`. The current benchmark feature is `cuda-ort` rather than the
stale `bench,cuda` spelling.

## Roofline arithmetic

The estimate uses effective H200 bandwidth
`B = 3.35e12 bytes/s` and the explicit weight-plus-KV model:

```text
bytes read/token = streamed weight bytes + average fp16 KV bytes read
roofline tok/s   = B / bytes read/token
KV/cached token  = layers * 2(K,V) * kv_heads * head_dim * sizeof(fp16)
```

Following the prior Joi accounting, streamed weight-side bytes are exact ONNX
initializer storage excluding the full fp16 embedding table, because decode
reads one embedding row:

```text
all initializer storage = 569,507,912 bytes
fp16 embedding table     = 311,164,928 bytes
streamed weights         = 258,342,984 bytes
KV/cached token          = 28 * 2 * 8 * 128 * 2
                         = 114,688 bytes
```

`profile_decode` has no steady-window skip. For the five-token prompt, average
cached context is `5 + (N - 1) / 2`: 68.5 tokens for 128 output tokens and
516.5 for 1024.

```text
128:  KV read  = 68.5 * 114,688 = 7,856,128 bytes
      total    = 258,342,984 + 7,856,128
               = 266,199,112 bytes/token
      roofline = 3.35e12 / 266,199,112 = 12,585 tok/s
      achieved = 197.49 / 12,585 = 1.57%

1024: KV read  = 516.5 * 114,688 = 59,236,352 bytes
      total    = 258,342,984 + 59,236,352
               = 317,579,336 bytes/token
      roofline = 3.35e12 / 317,579,336 = 10,549 tok/s
      achieved = 64.01 / 10,549 = 0.61%
```

The low percentages show that eager decode is dominated by orchestration,
binding, and kernel-launch costs rather than HBM bandwidth. At 1024 tokens the
as-exported `ort.bind_inputs` alone reaches 6.938 ms/token.

## Qwen2.5-0.5B comparison

A matched ORT CUDA eager control used the same binary, prompt, warmups, runs,
device-KV setting, and `CUDA_GRAPH=0`:

| Model/configuration | 128 tok/s | 1024 tok/s |
|---|---:|---:|
| Qwen3-0.6B as exported | **197.49** | **64.01** |
| Qwen3-0.6B corrected-metadata diagnostic | **429.95** | **381.49** |
| Qwen2.5-0.5B matched ORT eager control | **570.61** | **501.88** |
| Qwen2.5-0.5B supplied native/graph reference | **810.06** | **778.59** |

The exported Qwen3 package is 65.4%/87.2% slower than the matched Qwen2.5 ORT
control. Correcting metadata recovers 117.7% at 128 and 496.0% at 1024, leaving
Qwen3 24.7%/24.0% slower than Qwen2.5 under matched eager conditions. That
remaining difference cannot be assigned to QK-norm alone: Qwen3 also has 28
versus 24 layers, 8 versus 2 KV heads, larger attention head geometry, and 56
additional Q/K RMSNorm operations. QK-norm is functionally supported but is one
part of an approximately 24% matched eager architecture cost.

The supplied 810.06/778.59 tok/s Qwen2.5 numbers use native CUDA, whole-step
CUDA graph replay, and `--steady --decode-skip 8`; they are context, not an
apples-to-apples denominator for this ORT eager run.

## Coverage gap and exact commands

The graph and QK-norm operators are supported: CUDA load and coherent generation
pass. The gap is the exported native metadata contract:

```yaml
model:
  attention:
    type: grouped_query       # runtime shared-KV recognizer does not accept this alias
# kv_cache.native_dtype is absent
```

`shared_kv_buffer_len_from_metadata` accepts
`group_query_attention`, `grouped_query_attention`, or `gqa`, and also requires
a share-buffer-compatible KV dtype. Consequently, the supplied package selects
the growing `ZeroCopyRebind` path despite `ONNX_GENAI_DEVICE_KV=1`.

```bash
cd /home/justinchu/wt-joi-qwen3

export LD_LIBRARY_PATH=$(python3 -c \
  "import os,glob;print(':'.join(glob.glob(os.path.expanduser('~')+'/.conda/envs/*/lib/python*/site-packages/nvidia/*/lib')))"):/home/justinchu/onnx-genai/.ort-cuda-1.27/onnxruntime-linux-x64-1.27.0/lib:$LD_LIBRARY_PATH
export ORT_ROOT=/home/justinchu/onnx-genai/.ort-cuda-1.27/root
export LD_LIBRARY_PATH=$ORT_ROOT/lib:$LD_LIBRARY_PATH

cargo build --release -p onnx-genai-bench \
  --features cuda-ort --bin profile_decode

export CUDA_VISIBLE_DEVICES=0
export ONNX_GENAI_EP=cuda
export ONNX_GENAI_REQUIRE_CUDA=1
export ONNX_GENAI_DEVICE_KV=1
export ONNX_GENAI_CUDA_GRAPH=0
export ONNX_GENAI_PROFILE=1

./target/release/profile_decode \
  --model /home/justinchu/mobius/.scratch/qwen3-0.6b-int4-cuda \
  --tokens "$TOKENS" --warmups 2 --runs 3 --raw \
  --prompt "The capital of France is"
```

The command was run with `TOKENS=128` and `TOKENS=1024`, never concurrently.
