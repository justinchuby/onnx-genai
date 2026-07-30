### 2026-07-29: Large-model CUDA and weight-offload validation is blocked before decode
**By:** Mary

**What:** Validated the current `main` large-model path with Qwen3.5-35B-A3B
(MoE) first, then converted Qwen3.6-27B (dense) to MatMulNBits INT4 and attempted
the requested native-CUDA and ORT-CUDA profiler runs. No tokens were generated:
the MoE exporter does not emit the route-first/offload operator, the dense model
is missing required explicit I/O metadata and falls back from native CUDA, and
the pinned ORT-CUDA 1.27 runtime aborts while loading this Qwen3.5-family graph.
Consequently tok/s, decode coherence, and offload ON/OFF deltas are not
measurable on current `main`.

**Why:** This records the exact E2E gaps rather than reporting a component-level
weight-pager test as successful model offloading. The CUDA pager's isolated GPU
tests pass, but its own source still says executor/kernel consumption is
deferred, and issues #63, #82, and #87 remain open.

#### Revisions and hardware

- onnx-genai: `ba95c4c1` (`origin/main`, 2026-07-30 UTC)
- Möbius: `2227f5d` (`origin/main`)
- GPU: NVIDIA H200 143,771 MiB; GPUs 4-7 were idle
- ORT-CUDA requested by the project: 1.27.0
- Model cache initially contained only Qwen3.5-35B-A3B metadata. The full
  71.9 GB checkpoint was downloaded successfully in 2m36s.

#### Model/conversion results

| Model | Result | Artifact |
|---|---|---|
| Qwen3.5-35B-A3B MoE | Full checkpoint downloaded; graph-only conversion probe completed, but no usable INT4/offload graph can be produced on Möbius `main` | `/home/justinchu/mary-models/qwen3.5-35b-a3b-graph-probe.log` |
| Qwen3.6-27B dense | FP16 CUDA-target graph exported (53,826,551,808-byte external data); ORT RTN converted 497 MatMuls to block-32 INT4 MatMulNBits | `/home/justinchu/mary-models/qwen3.6-27b-int4-cuda/` (16 GB) |

The MoE graph probe produced 156,467 nodes, including 31,111 ordinary MatMuls,
30 LinearAttention nodes, zero QMoE nodes, zero `pkg.nxrt::BlockQuantizedMoE`
nodes, and zero MatMulNBits nodes. Post-export MatMul quantization cannot recover
route-first execution: it leaves the loop-over-all-256-experts topology intact.
This makes the requested 35B conversion intractable and, more importantly,
incapable of exercising expert paging.

The dense INT4 graph contains 3,144 top-level nodes, 497 MatMulNBits nodes,
zero ordinary MatMuls, and 48 LinearAttention nodes. ORT 1.28 CPU loaded it in
19.13 seconds, proving that the INT4 artifact is structurally loadable.

#### Requested benchmark table

| Model/backend | Offload | Peak VRAM | tok/s | Coherence | Result |
|---|---:|---:|---:|---|---|
| Qwen3.6-27B INT4 native CUDA | OFF | 1,495 MiB during failed load | N/A | N/A | Whole graph fell back to CPU because CUDA declines 96 FP16 ReduceSumSquare nodes; load then failed on ambiguous token input metadata |
| Qwen3.6-27B INT4 ORT-CUDA 1.27 | OFF | 8,415 MiB during failed load | N/A | N/A | ORT process aborted in `std::vector<NodeArg*>::operator[]` during session initialization |
| Qwen3.5-35B-A3B native/ORT | ON/OFF | N/A | N/A | N/A | No route-first/offload-capable model can be emitted |

Offloading ON/OFF cannot be honestly measured:

1. `ONNX_GENAI_WEIGHT_OFFLOAD=1` controls the CPU QMoE mmap/host-cache path,
   not the CUDA MatMulNBits model.
2. Lazy CUDA handles are recognized only at the
   `pkg.nxrt::BlockQuantizedMoE` boundary.
3. Möbius `main` does not emit `BlockQuantizedMoE`.
4. `crates/onnx-runtime-ep-cuda/src/weight_paging.rs` explicitly says binding
   the pager into executor `BlockQuantizedMoE` dispatch is deferred.

The isolated CUDA pager validation did pass:

```text
cargo test --release -p onnx-runtime-ep-cuda \
  --test weight_offload_gpu --features cuda -- --nocapture

2 passed; 0 failed
```

This proves byte-identical H2D paging, not live model offloading.

#### Bugs/root causes

1. **MoE export/offload boundary missing.** Unquantized Qwen3.5 MoE expands every
   expert into ordinary MatMuls. The fused QMoE path requires a GPTQ/AWQ
   quantization config before graph construction, while the available checkpoint
   is BF16. The runtime's lazy CUDA path requires a different
   `BlockQuantizedMoE` boundary that Möbius does not emit.
2. **Explicit metadata missing.** Generated `inference_metadata.yaml` contains
   attention/KV dimensions but no `model.io.token_input` or recurrent
   `state_pairs`. Native load reports three matching INT64 rank-2 inputs
   (`input_ids`, `attention_mask`, `position_ids`). This overlaps active issue
   #377/Möbius work; Benny's separate branch was not touched.
3. **Native CUDA coverage gap.** Qwen3.6's decomposed linear-attention graph has
   96 FP16 ReduceSumSquare nodes; CUDA accepts only Float32, and heterogeneous
   CUDA+CPU execution is unavailable, so the entire 27B graph falls back.
4. **ORT version incompatibility.** Both FP16 and INT4 exports abort with the
   project-pinned ORT-CUDA 1.27.0 during session initialization. The same INT4
   model loads successfully with ORT 1.28.0 CPU, so this is not introduced by
   INT4 quantization. Möbius `main` also now requires ORT >=1.28 in its relevant
   Attention compatibility test.

No code fix or PR was opened: a correct solution spans the exporter contract,
explicit metadata work already owned by Benny, CUDA LinearAttention primitive
coverage, and the ORT runtime pin. It is not a clean isolated patch.

#### Exact reproduction

```bash
# Isolated worktrees
git -C /home/justinchu/mobius worktree add \
  /home/justinchu/mary-mobius origin/main
git -C /home/justinchu/onnx-genai worktree add \
  -b squad/large-model-offload-validation \
  /home/justinchu/mary-onnx-genai origin/main

# Fetch the MoE checkpoint (71.9 GB)
/home/justinchu/.conda/envs/onnx/bin/hf download Qwen/Qwen3.5-35B-A3B

# Dense export used mobius.build with:
#   model=Qwen/Qwen3.6-27B
#   module_class=Qwen35CausalLMModel
#   task=hybrid-text-generation
#   dtype=f16
#   execution_provider=cuda
# followed by package.save() and write_onnx_genai_config().

# Fast block-32 RTN MatMulNBits conversion
PYTHONNOUSERSITE=1 /home/justinchu/.conda/envs/onnx/bin/python - <<'PY'
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
src = "/home/justinchu/mary-models/qwen3.6-27b-f16-cuda/model.onnx"
dst = "/home/justinchu/mary-models/qwen3.6-27b-int4-cuda/model.onnx"
q = MatMulNBitsQuantizer(src, bits=4, block_size=32, is_symmetric=True)
q.process()
q.model.save_model_to_file(dst, use_external_data_format=True)
PY

# Profiler build
cd /home/justinchu/mary-onnx-genai
source /home/justinchu/onnx-genai/.cudaenv.sh
export ONNX_GENAI_ORT_LIB="$ORT_ROOT/lib/libonnxruntime.so.1.27.0"
CUDA_VISIBLE_DEVICES=4 taskset -c 4 cargo build --release \
  -p onnx-genai-bench --bin profile_native \
  --features bench-native,bench-ort,cuda

# Native failure reproduction; replace native with ort for ORT-CUDA abort
CUDA_VISIBLE_DEVICES=4 taskset -c 4 ./target/release/profile_native \
  --model /home/justinchu/mary-models/qwen3.6-27b-int4-cuda \
  --ep cuda --backend native --steady --tokens 128 --warmups 2 --runs 3 \
  --decode-skip 8 \
  --prompt 'Explain what a transformer is in two sentences.'
```
