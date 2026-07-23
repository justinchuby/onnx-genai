---
name: profiling
description: How to profile the native CUDA/CPU EP with Nsight (ncu/nsys) and the built-in per-op timer. Read before profiling decode kernels.
---

# Profiling the native EP

Always `source /home/justinchu/onnx-genai/.cudaenv.sh` first (sets PATH, ORT libs, and the nvidia wheel libs on `LD_LIBRARY_PATH` so cuBLAS/cuDNN resolve). `ncu`/`nsys`/`nvcc` live in `/usr/local/cuda-13.3/bin`.

## The workload: profile_native

Build once, then run the steady-state decode loop:

```bash
cargo build --release -p onnx-genai-bench --features bench-native,cuda --bin profile_native
profile_native --model <model-dir> --ep cuda --steady --warmups 1 --runs 3 --tokens 128
```

## Nsight Compute (ncu) — per-kernel counters

`RmProfilingAdminOnly=1` on this box, so **ncu needs sudo with the env forwarded** (a bare `sudo ncu` loses PATH/LD_LIBRARY_PATH → cublasLt.so.12 fails):

```bash
sudo -E env PATH="$PATH" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  ncu --graph-profiling node --set full -k regex:<kernel> \
  --launch-count N -o report <profile_native cmd...>
```

- **`--graph-profiling node` is mandatory** for Phi/Qwen/GLM-4 — they run a captured CUDA graph, so without it every captured kernel is hidden.
- Isolate a target kernel with `-k regex:matmul_nbits_gemv...` + `--launch-skip`/`--launch-count` (a decode step launches each layer's kernel many times).
- Key decode signal: M=1 GEMVs are memory-latency/issue-bound (~13–18% DRAM of peak), dominant stall = **Long Scoreboard** (global-load latency), not bandwidth. Raising occupancy alone rarely helps.

## Nsight Systems (nsys) — timeline / kernel mix

```bash
sudo -E env PATH="$PATH" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  nsys profile --cuda-graph-trace=node -o timeline <profile_native cmd...>
nsys stats --report cuda_gpu_kern_sum timeline.nsys-rep   # per-kernel % of decode
```

`--cuda-graph-trace=node` is the nsys equivalent of `--graph-profiling node`; without it captured kernels collapse into one opaque graph node. Use the kernel-sum report to find the dominant kernel (the % to attack).

## Per-op timing (no Nsight, CPU or CUDA)

```bash
ONNX_GENAI_PROFILE_OPS=1 profile_native --model <dir> --ep <cpu|cuda> --steady --runs 3 --tokens 128
```

`executor.rs` prints per-op-type total_ms/percent/calls per forward pass to stderr — fast way to find which op-type dominates before reaching for Nsight.

## Notes

- A separate CPU-bench team shares this host → timing variance. Report medians, caveat variance.
- Verify byte/near-identity after any kernel change; split-K reorders fp32 partials (near-equal, not bit-exact) — validate with tolerance tests.
