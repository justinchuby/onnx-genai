# Decision: CUDA fp16 TopK — kernel already merged (#445); added router-scale conformance parity

**Author:** Cohaagen (EP/runtime perf)
**Date:** 2026-08-03
**Branch/PR:** squad/cuda-fp16-topk

## Context
Qwen3.6-35B-A3B (and any `dense_fallback` MoE decoder) has a 256-expert/top-8 router
whose gate `TopK` runs in fp16. The CUDA EP previously rejected fp16 TopK
(`input 0 ('X') dtype Float16 unsupported; expected Float32`) for all 40 router
nodes, forcing a whole-session CPU fallback.

## Finding
The fp16 (and bf16) TopK kernel + claim gate were **already implemented and merged**
on `origin/main` via **PR #445 "feat(cuda): support fp16 router TopK"** (commit
`d2333664`). The kernel upcasts each compared value to f32 (`static_cast<float>`),
reuses the existing total-order `before()` compare (equivalent to `f32::total_cmp`),
and writes back the original raw fp16/bf16 element — so selection is byte-identical
to the CPU/ORT oracle (fp16→f32 widening is lossless; the round-trip preserves the
element, including sign-of-zero). The claim gate now uses
`require_one_of(input_dtypes, 0, CUDA_FLOAT_DTYPES, "X")`
(`CUDA_FLOAT_DTYPES = [Float32, Float16, BFloat16]`).

## What this PR adds (the remaining task-mandated gap)
PR #445's conformance test covered only a small final-axis ties case. This PR extends
`tests/indexing_gpu.rs` with the coverage the task explicitly required:
- **Router-scale byte-parity:** fp16 `[2,256]` top-8 (the real 35B router shape) with
  a `%37` tie-heavy pattern, asserting GPU == CPU byte-for-byte + distinct/in-range
  selected experts.
- **Non-final axis:** fp16 axis-0 oracled against the proven f32 GPU kernel.
- **Claim-regression guard** (`cuda_ep_claims_fp16_topk_router_nodes`): asserts the EP
  now CLAIMS fp16/bf16 TopK via `supports_op` (and still claims f32).

## Verification
- Decoder `qwen36-35b-a3b-artifacts/decoder/model.onnx`: **40 TopK nodes, all input0
  elem_type=10 (Float16)** — confirms the exact op gap.
- `cargo test -p onnx-runtime-ep-cuda --features cuda --test indexing_gpu`: 15/15 pass
  (incl. the 3 fp16 TopK cases) on H200.
- Coverage-of-coverage (`every_covered_op_has_a_conformance_entry`): pass.
- `cargo clippy -p onnx-runtime-ep-cuda --features cuda -- -D warnings`: clean;
  `cargo fmt --all --check`: clean.

## Pre-existing issue observed (OUT OF SCOPE — flagged, not fixed)
The **CPU EP** `TopK` writes **non-final-axis** outputs in push order
(`outer, inner, k`) whereas the **CUDA kernel and ONNX** use k-major
(`outer, k, inner`) — the layout asserted by the existing `topk_non_final_axes_*`
GPU test. They therefore disagree byte-for-byte for a non-final axis with `inner>1`
and `k>1`. This is **independent of dtype** (affects f32 identically) and predates
this work. Recommend a separate fix to `crates/onnx-runtime-ep-cpu/src/kernels/selection.rs`
(TopK) to place outputs by tensor strides. This PR sidesteps it by oracling the
fp16 non-final-axis case against the f32 GPU path.

## Scope note
The full 35B native decode remains blocked by native pipeline decode (GAP 3) and
rank-3 mRoPE positions — out of scope here. This work only proves the TopK op gap is
closed (EP claims the 40 fp16 router nodes; standalone fp16 TopK runs byte-correct on
CUDA at router scale).
