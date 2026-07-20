# MLAS int4 end-to-end decode benchmark

**Date:** 2026-07-20T05:20Z
**Host:** Intel(R) Xeon(R) Platinum 8480C (Sapphire Rapids)

## Result

The newly landed MLAS SQNBit path is **not an end-to-end decode throughput win**
for the tested Qwen2.5 0.5B int4 model. With otherwise identical binaries and
arguments, selecting MLAS approximately halved throughput.

| Model | New tokens | Baseline (SimdX86) | MLAS SQNBit | MLAS / baseline |
|---|---:|---:|---:|---:|
| Qwen2.5 0.5B int4 | 128 | 18.08 tok/s (55.308 ms/step) | 9.53 tok/s (104.963 ms/step) | **0.527x (-47.3%)** |
| Qwen2.5 0.5B int4 | 256 | 17.50 tok/s (57.144 ms/step) | 9.04 tok/s (110.645 ms/step) | **0.517x (-48.3%)** |

Each performance result used two warmups and three measured runs, performed
sequentially to avoid benchmark contention.

## Correctness

Both modes produced coherent text and answered with Paris, but token output was
not identical. The difference is consistent with the different quantized
compute paths (`accuracy_level=4` selects MLAS CompInt8), and neither output was
garbage.

- **Baseline:** “Paris. It is the largest city in the country and the capital
  of France. It is also the most populous city in the country...”
- **MLAS:** “Paris. It is the largest city in the world and the most populous
  city in the world... The answer is **Paris**...”

The secondary GLM artifact had both required files, but was not runnable with
the required prompt: tokenization failed with `WordLevel error: Missing [UNK]
token from the vocabulary`.

## Path confirmation

The profiler was built with the bench crate's `mlas` feature, which forwards
`onnx-genai-engine/mlas` and `onnx-runtime-ep-cpu/mlas`. ONNX inspection found
121 `com.microsoft::MatMulNBits` nodes in Qwen, all `bits=4`,
`block_size=32`, `accuracy_level=4`, with no `g_idx`. Therefore the baseline
uses auto-detected SimdX86, while `NXRT_CPU_GEMM_BACKEND=mlas` satisfies the
SQNBit dispatch gate. `RUST_LOG=debug` emitted no additional kernel-selection
line, so confirmation is from the build/runtime gates and graph attributes.

## Commands

```bash
cargo build --release -p onnx-genai-bench --features mlas --bin profile_native

env -u NXRT_CPU_GEMM_BACKEND ./target/release/profile_native \
  --model /home/justinchu/qwen2.5-0.5b-int4-onnx \
  --tokens 128 --warmups 2 --runs 3 --ep cpu \
  --prompt "The capital of France is"

NXRT_CPU_GEMM_BACKEND=mlas ./target/release/profile_native \
  --model /home/justinchu/qwen2.5-0.5b-int4-onnx \
  --tokens 128 --warmups 2 --runs 3 --ep cpu \
  --prompt "The capital of France is"

env -u NXRT_CPU_GEMM_BACKEND ./target/release/profile_native \
  --model /home/justinchu/qwen2.5-0.5b-int4-onnx \
  --tokens 256 --warmups 2 --runs 3 --ep cpu \
  --prompt "The capital of France is"

NXRT_CPU_GEMM_BACKEND=mlas ./target/release/profile_native \
  --model /home/justinchu/qwen2.5-0.5b-int4-onnx \
  --tokens 256 --warmups 2 --runs 3 --ep cpu \
  --prompt "The capital of France is"

env -u NXRT_CPU_GEMM_BACKEND ./target/release/profile_native \
  --model /home/justinchu/glm-e2e-artifacts/glm-5.2-tiny-q4 \
  --tokens 128 --warmups 2 --runs 3 --ep cpu \
  --prompt "The capital of France is"

NXRT_CPU_GEMM_BACKEND=mlas ./target/release/profile_native \
  --model /home/justinchu/glm-e2e-artifacts/glm-5.2-tiny-q4 \
  --tokens 128 --warmups 2 --runs 3 --ep cpu \
  --prompt "The capital of France is"
```

Generated token IDs were decoded with Python's `tokenizers.Tokenizer` using the
model's `tokenizer.json`.

## Caveat and interpretation

Autoregressive decode is dominated by small-`M` GEMV-like work. The existing
hand-written int4 path is specialized for `M=1`, while SQNBit's strongest
microbenchmark wins are expected at prefill or larger `M`. These results do not
invalidate larger-`M` SQNBit gains, but they show that enabling MLAS globally is
currently a substantial regression for end-to-end single-sequence decode on
this small model.
