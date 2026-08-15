# Llama-3.2-1B native decode: fp16 tied-head fusion (2026-07-23)

**Author:** Roy (CUDA/EP performance) · **Branch:** `squad/roy-lmhead-fusion`
**Device:** NVIDIA H200 (143 GB HBM3e, ~3.35 TB/s) · CUDA EP, native decode
**Bench:** `profile_native --ep cuda --steady --warmups 1` (decode_skip 8)

## Problem

Llama-3.2-1B-Instruct native decode was **97 tok/s** vs ORT **589 tok/s** — a ~6×
gap — even with the device-KV / CUDA-graph / GQA shared-buffer fast path fully
enabled. The model has a **tied embedding / fp16 output head**: the fp16
`[vocab, hidden]` embedding weight is `Gather`-ed for input embeddings and, for
the LM head, `Transpose`-d to `[hidden, vocab]` and fed to a **dense fp16
`MatMul`** every decode step. Qwen2.5/Qwen3 don't hit this because their lm_head
is a quantized `MatMulNBits`.

## Root cause (native decode trace, @32 tokens, eager per-op timing)

| op            | total | n  | avg/step | note |
|---------------|------:|---:|---------:|------|
| **Transpose** | 311.3 ms | 32 | **9.73 ms** | re-transpose of the ~525 MB fp16 embed weight, every step |
| **MatMul**    |  66.7 ms | 32 | **2.08 ms** | dense fp16 `[1,2048]×[2048,128256]` GEMV, cuBLASLt, non-capturable |
| GroupQueryAttention | 54.3 ms | 32 | 1.70 ms | |
| MatMulNBits (×14/step) | 27.4 ms | 224 | 0.12 ms | |

The per-step `Transpose` over a half-GB constant dominated; the dense fp16 GEMV
was second. Both operate on a **constant** weight, re-doing work every token.

## Fixes — two generic, pattern-based EP-internal optimizations

Both live in the CUDA EP (`onnx-runtime-ep-cuda`) and are gated by **topology +
tensor roles + dtype/shape**, never by model name (RULES.md §2 / §2.1).

### 1. Constant-initializer `Transpose` folding (`CudaFoldConstantTranspose`)

A new EP optimization pass. Any `Transpose` in the default/`ai.onnx` domain whose
single input is a **producer-less graph initializer** with a whole-byte element
type is materialized once, at EP claim/compile time, into a pre-transposed inline
initializer; consumers are rewired to the constant and the node is deleted. The
permutation is applied byte-wise over the raw little-endian element bytes, so it
is correct for any rank / `perm` and any whole-byte dtype. Sub-byte packed weights
(int4/…) and non-constant inputs are skipped. The original initializer is left
intact for its other consumers (e.g. the tied-weight `Gather`).

Effect: the per-step transpose disappears entirely.

### 2. Dense fp16 M==1 GEMV fast path (in the `MatMul` kernel)

`MatMulKernel::run` now routes a **dense fp16, M==1, single-matrix** MatMul to a
dedicated NVRTC GEMV kernel (`matmul_dense_gemv_f16`, portable across SMs — it is
compiled to the device's own architecture). One thread owns one output column, so
consecutive threads read consecutive `B[k, col]` fp16 values — a fully coalesced
streaming pass over `B` at ≈ HBM roofline. Activation is staged in shared memory
per K-tile (bounded to `blockDim.x` floats for any K); accumulation is fp32 to
match the cuBLASLt path. The gate is purely structural (dtype + M==1 + no batch),
never a model dimension, so it fires for any dense fp16 head.

Bonus: unlike the cuBLASLt path (per-call workspace alloc/free + heuristic query,
`CaptureSupport::Unsupported`), the GEMV needs no allocation or synchronization,
so it is **capture-safe** and folds into the decode CUDA graph (`capture_status:
captured`).

## Results — Llama-3.2-1B-Instruct (Q4_K_M, fp16 tied head)

| stage | @128 tok/s | @1024 tok/s | decode ms/step @128 |
|-------|-----------:|------------:|--------------------:|
| baseline (origin/main) | **97.5** | ~97 | 10.26 |
| + Transpose fold       | 409.4 | — | 2.44 |
| + fp16 GEMV fast path  | **449.1** | **438.3** | 2.23 |

**Net: 97 → 449 tok/s @128 (4.6×), 438 tok/s @1024.** Greedy token IDs are
byte-identical to baseline at every stage (coherent output — the model emits valid
code/text). Remaining gap to ORT (589) is in GQA / MatMulNBits / norm, not the head.

## No regression — Qwen2.5-0.5B (int4, quantized `MatMulNBits` head)

Qwen's graph has **no `Transpose` and no dense `MatMul`** (verified via trace), so
neither optimization can fire. Measured identical, same command / machine:

| model | @128 baseline → branch | @1024 baseline → branch |
|-------|-----------------------:|------------------------:|
| qwen2.5-0.5b-int4-onnx-native | 314.0 → 313.5 | 84.89 → 84.90 |

(The ~314/85 figures are this machine's numbers; the point is **no regression**:
within run-to-run noise, and structurally the code paths are inert for Qwen.)

## Tests

* `onnx-runtime-ep-cuda` unit (`optimizer.rs`): `folds_constant_transpose_into_initializer`,
  `folds_constant_transpose_default_perm`, `folds_rank3_constant_transpose`,
  `leaves_transpose_of_non_constant`, `leaves_sub_byte_constant_transpose`.
* GPU integration (`tests/matmul_gpu.rs`): `matmul_f16_gemv_on_gpu_matches_cpu_reference`
  — non-square GEMV (K=259, N=300) vs CPU reference, and asserts capture support.

## Repro

```
export LD_LIBRARY_PATH=<cuda libs>:$LD_LIBRARY_PATH
cargo build --release -p onnx-genai-bench --features bench-native,cuda --bin profile_native
./target/release/profile_native --model <llama-q4km-dir> --ep cuda --tokens 128 --steady --warmups 1 --runs 3
```
