# Resch KAI-style packed SDOT kernel follow-up

Date: 2026-07-29
Branch: qwen3-perf-followups
Owner: Resch
Scope: `crates/onnx-runtime-ep-cpu/src/kernels/matmul_nbits.rs` only

## Implementation

Added a correctness-first, KleidiAI-inspired ARM64 dotprod decode path for M=1 `MatMulNBits`:

- `qsi4` (`bits=4`) and `qsi8` (`bits=8`) share `PackedKaiSdotWeight`.
- Prepack keeps quantized RHS compact:
  - qsi4 stores two centered nibbles per byte.
  - qsi8 stores centered signed bytes.
  - layout is `[ceil(N/4), k_blocks, block_size/4, 4 outputs, payload]`.
- Prepack also stores per-output/per-block scale, RHS sums, and zero-point offsets.
- Added `qai8dxp`-style activation quantization once per decode row with row/block sums for asymmetric correction.
- Added scalar reference and aarch64 `dotprod` implementation; non-aarch64 remains on existing fallbacks.
- Dispatch is still gated by `ONNX_GENAI_CPU_ARM64_INT4_DIRECT` outside tests because perf did not beat ORT.

## Correctness / reachability validation

Passed:

- `cargo check -p onnx-runtime-ep-cpu --tests --quiet`
- `cargo clippy -p onnx-runtime-ep-cpu --tests --quiet -- -D warnings`
- `cargo test -p onnx-runtime-ep-cpu kai_sdot --quiet`
- `cargo test -p onnx-runtime-ep-cpu arm64_kai --quiet`
- `cargo test -p onnx-runtime-ep-cpu matmulnbits_arm64_kai --quiet`
- `cargo test -p onnx-runtime-ep-cpu matmulnbits_8bit --quiet`
- `cargo test -p onnx-runtime-ep-cpu n16 --quiet`
- `cargo test -p onnx-runtime-ep-cpu matmulnbits --quiet`

Coverage includes qsi4/qsi8 block128 asymmetric zero-points, Qwen-shaped N widths/tails, and reachability proving real eligible M=1 nodes select the KAI-style cache in tests.

## Full-model benchmark

Command pattern:

```powershell
$env:ONNX_GENAI_CPU_ARM64_INT4_DIRECT='1'
target\release\profile_native.exe --model <model_dir> --backend native --steady --warmups 1 --runs 5 --tokens 128
```

Median steady decode:

| Model | Roofline | Native KAI-gated | % roofline | ORT | ORT % roofline | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| qwen3-0.6b CPU-4 | 211 tok/s | 71.31 tok/s | 33.8% | 105.68 tok/s | 50.1% | beats gated-off native baseline but not ORT |
| qwen2.5-0.5b CPU-4 | 344 tok/s | 82.41 tok/s | 24.0% | 184.48 tok/s | 53.6% | not competitive |
| qwen3-1.7b CPU-2 | 76.5 tok/s | 25.48 tok/s | 33.3% | 49.52 tok/s | 64.7% | not competitive |

Baseline native with the gate off on qwen3-0.6b measured 57.06 tok/s, so this is real progress (+25%) but not the requested ORT win. The honest gate remains opt-in.

## Diagnosis

The implementation removed the worst N16 problem for qsi4 (no full int8 RHS expansion) and added the missing qsi8 direct path, but the Rust/NEON loop is not yet close to KleidiAI's instruction density:

1. qsi4 still pays too much unpack overhead per K4 group. KleidiAI amortizes qsi4c32p unpack across a fixed 4-output/32-K subblock with hand-scheduled loads and SDOTs.
2. The hot path lacks a hand-written 32-K microkernel with stable register allocation, prefetch, and software pipelining. LLVM does not reliably produce the same schedule from the Rust intrinsics loop.
3. qsi8 path correctness works, but qA8dxp quantization plus block correction overhead is not yet amortized enough; ORT/MLAS is still doing fewer instructions per weight byte.

## Recommendation

Keep this commit as the correctness/reachability milestone and ask Luba to turn the inner loop into an assembly/intrinsics microkernel before enabling by default:

- exact tile: M=1, N=4 or N=8, K subblock=32, block128 outer loop;
- qsi4: load 16 packed bytes per output per K32, unpack low/high nibbles to signed int8 in vectors, immediately SDOT with prepacked qA8dxp bytes;
- qsi8: load signed int8 RHS directly and use the same accumulator/dequant skeleton;
- maintain 4 or 8 int32 accumulators, fuse zp corrections once per block, then f32 scale once per block;
- add prefetch for next N tile/K block and split N tiles across decode_affinity threads.

This should be portable to Snapdragon and Apple NEON+dotprod. Apple Silicon should use this NEON path where Accelerate cannot cover quantized decode; AMX/Accelerate routes must remain untouched.
