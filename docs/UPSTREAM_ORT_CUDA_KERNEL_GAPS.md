# Upstream ORT CUDA Kernel Gap Analysis

**Author:** Batty (Engine Dev)
**Date:** 2026-08-11
**Branch:** `squad/upstream-ort-contrib-plan`
**Purpose:** Inventory of nxrt CUDA EP optimizations vs. upstream ORT, ranked by upstreamability.

---

## 1. Inventory of Our Meaningfully Optimized CUDA Kernels

| # | Feature / Kernel | Location | Optimization |
|---|---|---|---|
| 1 | **MatMulNBits int4/int8 decode GEMV** | `matmul_nbits.rs` | Block-128-specialized GEMV paths, int8 split-K for grid-starved down projections, accuracy_level=4 quantize-activation-then-GEMV | 
| 2 | **MatMulNBits SM-fill grid tuning** | `matmul_nbits.rs` (#148) | `down_tpl<COLS>` grid-fill heuristic splits CTAs based on SM count to avoid grid-starvation on H200 |
| 3 | **GQA decode split-K (f32)** | `gqa_decode.rs` | Up-to-16-CTA split-K per query head with online-softmax merge; capture-safe (no sync, fixed scratch) |
| 4 | **GQA decode flash-decode (fp16)** | `gqa_decode_fp16.rs` | Multi-warp fp16 flash-decode with up to 16 splits, even head_dim ≤ 256, fp32 accumulators |
| 5 | **QMoE (com.microsoft)** | `qmoe.rs`, `qmoe_gemm.rs`, `qmoe_grouping.rs` | Block-quantized affine int4 MoE with host-free top-k routing, fused SwiGLU activation, per-route GEMV decode + tiled grouped GEMM prefill |
| 6 | **BlockQuantizedMoE (pkg.nxrt)** | `block_quantized_moe.rs` | GGUF native-format MoE (mxfp4, iq4_nl, iq4_xs, iq2_xxs, etc.); host-free routing, block-cooperative parallel top-k |
| 7 | **QMoE router parallelization** | `block_quantized_moe.rs` / decisions | Block-cooperative reduction with total-order tie rule, ~62 tok/s on 35B-A3B (was serial) |
| 8 | **LinearAttention (Gated DeltaNet)** | `linear_attention.rs` | Column-parallel f32 recurrent state scan; capture-safe; hybrid Mamba/attention support |
| 9 | **CompressedSparseAttention (CSA)** | `compressed_sparse_attention.rs` | DeepSeek-V4-Flash / GLM-5.2 ratio-4/128 hybrid FP8 attention; capture-safe |
| 10 | **Device argmax** | `device_argmax.rs` | Allocation-free two-pass reduction; stays entirely on device; greedy sampling without D2H |
| 11 | **CUDA graph capture system** | Engine-level + per-kernel `subgraph_graph_capturable` | Per-op capture eligibility classification, VMM remap across capture, symbolic-dim pinning |
| 12 | **VMM weight management** | Engine-level (`memory_governor`) | Physical granule pooling, mapped growth grants, committed-granule admission, eviction |
| 13 | **Paged/tiered KV cache** | Engine-level | Device KV growth counters, KV tier migration, host KV page store |
| 14 | **RotaryEmbedding (opset 23)** | `rotary_embedding.rs` | Both GPT-NeoX and GPT-J conventions; position_ids gather; LongRoPE fully on-device |
| 15 | **Flash attention (prefill)** | `flash_attention.rs` | NVRTC tiled online-softmax, 8-query-row tile, no [B,H,Sq,Sk] allocation |
| 16 | **Packed varlen attention** | `packed_varlen_attention.rs` | Variable-length packed sequences for continuous batching |

---

## 2. Upstream Comparison

### 2.1 MatMulNBits int4/int8 GEMV (#1, #2)

**Upstream state:** ORT has `matmul_4bits.cu`, `matmul_8bits.cu`, and the `fpA_intB_gemv` path in `contrib_ops/cuda/quantization/` and `contrib_ops/cuda/llm/fpA_intB_gemv/`. These support int4 block-quantized GEMM/GEMV.

**Gap:** Our specializations include:
- Block-128-folded shift optimization (eliminates division/modulo per weight)
- SM-count-based grid-fill heuristic (#148: +10.5% on Qwen 1.5B, +2.1% on 7B)
- accuracy_level=4 quantize-activation-to-int8 GEMV for arbitrary block sizes
- int8 split-K for grid-starved down-projections

**Evidence:** ORT issue [#23004](https://github.com/microsoft/onnxruntime/issues/23004) tracks MatMulNBits performance. Issue [#29691](https://github.com/microsoft/onnxruntime/issues/29691) shows `fpA_intB` path limitations. Our scoreboard shows native 1.385–1.605× over ORT on H200 for int4 models (2026-07-25, `docs/benchmarks/2026-07-25-post148-native-vs-ort-scorecard.md`).

**Verdict: GAP (partial)** — ORT has int4 GEMV but lacks our block-128 specializations, grid-fill tuning, and accuracy_level=4 blockwise GEMV.

### 2.2 GQA Decode Split-K Attention (#3, #4)

**Upstream state:** ORT has `group_query_attention_impl.cu` and `paged_attention_impl.cu`. The standard GQA uses flash attention for prefill and a simpler decode path.

**Gap:** Our split-K implementation exposes parallelism across up to 16 CTAs per query head for decode (Sq=1), keeping latency flat over long contexts. Supports both f32 and fp16 with fp32 accumulators. The fp16 path handles even head_dim ≤ 256.

**Evidence:** Scoreboard shows GQA contributes ~0.406 ms/token on H200 (from `docs/benchmarks/2026-07-23-ort-vs-native-cuda-scoreboard.md` Nsight trace). Issue [#28352](https://github.com/microsoft/onnxruntime/issues/28352) discusses ORT ONNX Attention vs Contrib GQA performance gaps. Issue [#29783](https://github.com/microsoft/onnxruntime/issues/29783) requests quantized KV + CUDA graph capture for GQA.

**Verdict: GAP** — ORT's GQA decode path exists but appears to lack split-K parallelism and capture-safe guarantees. However: unverified whether ORT's latest contrib GQA (which calls into flash_attn or memory-efficient attention) already has a similar split-KV decode tier.

### 2.3 QMoE / Block-Quantized MoE (#5, #6, #7)

**Upstream state:** ORT has `qmoe_kernels.cu` in `contrib_ops/cuda/moe/`. Also `moe_kernels.cu` and `moe_gemv.cu` in `contrib_ops/cuda/llm/moe_gemm/`.

**Gap:**
- Our BlockQuantizedMoE supports GGUF native formats (mxfp4, iq4_nl, iq4_xs, iq2_xxs, iq3_xxs, iq2_xs, iq2_s, iq3_s, iq1_s, iq1_m) — ORT's QMoE supports only the standard int4 affine layout.
- Block-cooperative parallel top-k routing (62 tok/s on 35B-A3B, was serial)
- Issue [#29035](https://github.com/microsoft/onnxruntime/issues/29035) (closed) requested block-wise quantized expert weights — suggests upstream may have added partial support.
- Issue [#28163](https://github.com/microsoft/onnxruntime/issues/28163) requests 2-bit support (open).
- Issue [#28987](https://github.com/microsoft/onnxruntime/issues/28987) tracks 35B-A3B throughput optimization (open).

**Verdict: PARTIAL** — ORT has basic QMoE int4. Lacks GGUF sub-4-bit formats and parallel routing. The parallel routing kernel is self-contained and upstreamable.

### 2.4 LinearAttention / Gated DeltaNet (#8)

**Upstream state:** ORT has `linear_attention_impl.cu` and `linear_attention_gates_impl.cu` in `contrib_ops/cuda/bert/`.

**Verdict: ALREADY-COVERED** — ORT already has a CUDA LinearAttention implementation. Our port is a faithful reimplementation (the code says "Faithful CUDA port of the CPU EP kernel… itself a port of ORT's contrib_ops/cpu/bert/linear_attention.cc"). Likely no gap here unless our parallelization is superior (unverified).

### 2.5 CompressedSparseAttention / DeepSeek MLA (#9)

**Upstream state:** No `CompressedSparseAttention` found in ORT. Issue [#23925](https://github.com/microsoft/onnxruntime/issues/23925) requests MLA support (open). No code evidence of DeepSeek MLA/CSA in ORT CUDA EP.

**Gap:** Our CSA supports ratio-4 and ratio-128 hybrid FP8 paths for DeepSeek-V4-Flash / GLM-5.2.

**Verdict: GAP** — but this is a `pkg.nxrt` custom op deeply coupled to our engine's KV management (past/present threading through device-resident state, VMM-managed KV). Not easily upstreamable without ORT adopting the same KV pattern.

### 2.6 Device Argmax (#10)

**Upstream state:** No dedicated device argmax in ORT's CUDA EP (search returned only generic `reduction_functions.cu`). ORT's greedy search likely uses `ArgMax` as a standard ONNX op with a generic reduction kernel, or does D2H for sampling.

**Verdict: GAP** — Self-contained kernel, but only useful if ORT's generate loop keeps logits on device. Currently ORT GenAI copies logits to host for sampling. The kernel is trivial (two-pass block reduction) and the value is in the *system design* (staying on-device), not the kernel itself.

### 2.7 CUDA Graph Capture System (#11)

**Upstream state:** ORT supports CUDA graph capture at the session level (provider option `enable_cuda_graph`). Issue [#29783](https://github.com/microsoft/onnxruntime/issues/29783) requests CUDA graph + shared buffer + quantized KV.

**Gap:** Our system does per-op capture eligibility classification, symbolic-dim pinning for growing dimensions, VMM remap across capture boundaries, multi-segment capture with eager fallback per seam. ORT's approach is whole-session capture with static shapes.

**Verdict: GAP but RUNTIME-LEVEL** — This is an architectural system, not a portable kernel. Cannot be contributed as a kernel patch.

### 2.8 VMM Weight Management / Paged KV (#12, #13)

**Upstream state:** No VMM-based weight paging or tiered KV cache in ORT CUDA EP.

**Verdict: GAP but RUNTIME-LEVEL** — Deeply coupled to nxrt's memory governor, scheduler, and decode loop. Not a kernel contribution.

### 2.9 RotaryEmbedding (#14)

**Upstream state:** ORT has `rotary_embedding_impl.cu` in both `contrib_ops/cuda/bert/` and `core/providers/cuda/llm/`. Also `gemma_rotary_emb_impl.cu`.

**Verdict: ALREADY-COVERED** — ORT has comprehensive RoPE CUDA support.

### 2.10 Flash Attention (prefill) (#15)

**Upstream state:** ORT integrates flash_attn2 and memory-efficient attention from xFormers. They also have the standard `Attention` CUDA kernel.

**Verdict: ALREADY-COVERED** — ORT uses production flash attention implementations (flash_attn2, cuDNN fused attention). Our NVRTC tiled kernel is a simpler fallback; upstream's is likely better.

### 2.11 Packed Varlen Attention (#16)

**Upstream state:** ORT's flash attention integration includes variable-length support via flash_attn's `flash_attn_varlen_func`.

**Verdict: ALREADY-COVERED**

---

## 3. Critical Split: KERNEL-LEVEL vs RUNTIME-LEVEL

### KERNEL-LEVEL (self-contained, upstreamable)

| Candidate | Entanglement with nxrt | Notes |
|---|---|---|
| MatMulNBits block-128 GEMV specialization | None — pure CUDA kernel with standard inputs | Can be ported as a new dispatch path in ORT's `matmul_4bits.cu` |
| MatMulNBits SM-fill grid tuning (#148) | Minimal — uses SM count at launch | Standard CUDA occupancy optimization |
| MatMulNBits accuracy_level=4 blockwise GEMV | None — self-contained quantize + GEMV | New dispatch path for ORT's MatMulNBits |
| GQA decode split-K (f32 and fp16) | Minimal — needs pre-allocated fixed scratch | New decode tier for ORT's GQA. Capture-safety is a bonus |
| QMoE parallel routing kernel | None — pure block-cooperative reduction | Drop-in improvement to ORT's `qmoe_kernels.cu` routing |
| Device argmax | None — pure two-pass reduction | Trivial kernel; value is systemic |

### RUNTIME-LEVEL (entangled with nxrt, not directly upstreamable)

| Feature | Entanglement |
|---|---|
| CUDA graph capture system (per-op classification, symbolic pinning, VMM remap) | Scheduler, executor, symbol table, VMM governor |
| VMM weight paging (granule pooling, eviction, mapped growth) | Memory governor, weight lifecycle, session management |
| Paged/tiered KV cache (device growth counters, tier migration, host page store) | KV manager, scheduler preemption, continuous batching loop |
| CompressedSparseAttention (DeepSeek MLA) | Custom `pkg.nxrt` op domain, KV threading pattern, FP8 residency |
| BlockQuantizedMoE GGUF formats | Custom op domain, `MemoryRole` integration, block format registry |
| LinearAttention capture lane / hybrid cache guard | Executor capture state machine, `has_recurrent_state()` contract |

---

## 4. Ranked Shortlist of Upstreamable Candidates

| Rank | Candidate | Model-level Impact | Generality | Portability | Impl Cost | Upstream Acceptance | Score |
|---:|---|---|---|---|---|---|---|
| 1 | **MatMulNBits int4 block-128 GEMV** | HIGH — decode is GEMV-bound; +36–60% on 0.5B–1.5B models | All int4 block-128 models (majority) | HIGH — no nxrt deps | MEDIUM — new kernel entry + dispatch | HIGH — addresses [#23004](https://github.com/microsoft/onnxruntime/issues/23004) | ★★★★★ |
| 2 | **MatMulNBits SM-fill grid tuning** | MEDIUM — +2–10% on H200 depending on model size | Multi-SM GPUs (H100/H200/B200) | HIGH — SM-count query is standard CUDA | LOW — grid calculation change | HIGH — non-controversial perf fix | ★★★★☆ |
| 3 | **GQA decode split-K (fp16)** | MEDIUM — keeps decode latency flat at long context | All GQA/MQA models, fp16 | MEDIUM — needs scratch allocation convention | MEDIUM — two-kernel design | MEDIUM — ORT may prefer cuDNN path | ★★★★☆ |
| 4 | **QMoE parallel routing** | MEDIUM — +30% on MoE decode (62 vs ~47 tok/s) | All QMoE models | HIGH — pure CUDA reduction | LOW — routing kernel replacement | HIGH — directly improves [#28987](https://github.com/microsoft/onnxruntime/issues/28987) | ★★★★☆ |
| 5 | **MatMulNBits accuracy_level=4 blockwise GEMV** | MEDIUM — enables int8-quantized-activation GEMV at any block size | int4 models with accuracy_level=4 | HIGH — self-contained | MEDIUM — new GEMV + quantize kernel | MEDIUM — niche use case | ★★★☆☆ |

---

## 5. Per-Candidate Porting Notes

### Candidate 1: MatMulNBits int4 block-128 GEMV

**ORT target:** `onnxruntime/contrib_ops/cuda/quantization/matmul_4bits.cu` and `matmul_nbits.cuh`
**Rewrite needed:**
- Replace nxrt's NVRTC-compiled PTX with a static `.cu` kernel (ORT uses compile-time templates)
- Adapt to ORT's `OpKernelContext` / `Stream` / `IAllocator` instead of nxrt's `CudaRuntime`
- Use ORT's `CudaKernel` base class, `OrtMutex`, and thread-safe stream
- Register via `ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME` macro
- cuBLAS handle from ORT's `CublasHandle()` provider

**Tests upstream expects:** Unit tests in `onnxruntime/test/contrib_ops/` with golden values; `onnxruntime_test_all` must pass. Performance not gated at PR time but expected in description.

**nxrt deps to sever:** `CudaRuntime` (module loading, stream), `cudarc` crate (replace with raw CUDA driver API or ORT's existing `CudaCall` wrappers), NVRTC JIT (replace with compiled .cu).

### Candidate 2: MatMulNBits SM-fill grid tuning

**ORT target:** Same as #1; the grid calculation in ORT's existing GEMV dispatch.
**Rewrite needed:** Minimal — add `cudaGetDeviceProperties` or `cuDeviceGetAttribute` for SM count; adjust grid.x calculation. Possibly a compile-time template parameter or runtime branch.
**Tests:** Existing MatMulNBits tests should pass unchanged (numerics identical); add a benchmark note.

### Candidate 3: GQA decode split-K (fp16)

**ORT target:** `onnxruntime/contrib_ops/cuda/bert/group_query_attention_impl.cu`
**Rewrite needed:**
- Port split-K kernel + merge kernel as static `.cu`
- Integrate with ORT's GQA op, which has different KV buffer semantics (shared past/present buffer, seqlens_k tensor)
- Scratch allocation via ORT's `GetScratchBuffer<T>` instead of module-global fixed allocation
- Must respect ORT's existing causal mask / seqlens / rotary conventions

**Tests:** `test/contrib_ops/bert/attention_test.cc` suite; add Sq=1 decode-specific cases.

**Complexity:** MEDIUM-HIGH — ORT's GQA has complex mode selection (flash/memory-efficient/math/paged). Adding a split-K decode tier requires careful routing.

### Candidate 4: QMoE parallel routing

**ORT target:** `onnxruntime/contrib_ops/cuda/moe/qmoe_kernels.cu`
**Rewrite needed:**
- Replace serial top-k with block-cooperative reduction preserving total-order tie rule
- Match ORT's routing interface (softmax → top-k → scatter indices)
- Standard `.cu` kernel, no NVRTC

**Tests:** Add expert-selection golden tests ensuring deterministic tie-breaking.

### Candidate 5: MatMulNBits accuracy_level=4 blockwise GEMV

**ORT target:** Extend ORT's existing accuracy_level=4 path (which uses `fpA_intB_gemv`)
**Rewrite needed:** Port the two-step (quantize-activation → int8 GEMV) pattern for general block sizes. ORT's `fpA_intB_gemv` already does something similar but with restrictions ([#29691](https://github.com/microsoft/onnxruntime/issues/29691)).

---

## 6. Explicit Non-Candidates

| Feature | Reason |
|---|---|
| **CUDA graph capture system** | Architectural — per-op eligibility, symbol pinning, multi-segment capture are deeply woven into nxrt's executor and scheduler. ORT has its own (simpler) CUDA graph mode. |
| **VMM weight paging** | Architectural — requires memory governor, physical granule pooling, eviction policy. ORT uses standard allocators. |
| **Paged/tiered KV cache** | Architectural — coupled to nxrt scheduler, preemption, continuous batching. ORT's KV is session-owned. |
| **CompressedSparseAttention** | Custom domain (`pkg.nxrt`), FP8 hybrid paths coupled to residency management. Also unclear upstream demand (DeepSeek-specific). |
| **BlockQuantizedMoE (GGUF formats)** | Custom domain, memory-governor integration, GGUF-only format. ORT would need to adopt GGUF or map formats. |
| **Device argmax** | Kernel is trivial; value is systemic (on-device decode loop). ORT GenAI does host sampling. |
| **LinearAttention** | Already covered upstream. Our version is explicitly "a faithful port." |
| **Flash attention (prefill)** | ORT uses flash_attn2/cuDNN — production-grade, likely faster than our tiled NVRTC kernel. |

---

## 7. Hardware Requirements for Validation

| Candidate | Minimum GPU | Memory | Arch Notes |
|---|---|---|---|
| MatMulNBits block-128 GEMV | Any CUDA GPU with sm_70+ | 8 GB (for 7B int4) | Test on sm_70 (V100), sm_80 (A100), sm_90 (H100/H200) to confirm generality |
| SM-fill grid tuning | Multi-SM GPU (≥80 SMs) | Same | Benefit is proportional to SM count; test on H100 (132 SMs) and A100 (108 SMs) |
| GQA decode split-K | Any sm_70+ | 16 GB (long context) | Test with seq_len=2048, 8192, 32768 to show latency flatness |
| QMoE parallel routing | Any sm_70+ | 48+ GB (35B model) | Must test with 35B-A3B artifact for the MoE decode path |
| accuracy_level=4 blockwise | Any sm_70+ | 8 GB | Correctness is the priority |

**None of the above can be validated on this host (no GPU).**

---

## 8. Open Questions for Justin

1. **ORT's `fpA_intB_gemv` limitations:** Issue #29691 shows ORT's path crashes on small-N prefill. Is their decode GEMV path similarly limited, or does it already handle the shapes we optimize? Need to read `fpA_intB_gemv.cu` in detail.

2. **ORT's GQA decode tier:** Issue #29714 (closed) added a cuDNN SDPA decode tier. Does this supersede the need for a custom split-K kernel, or is there still a gap for non-cuDNN paths (older GPUs, non-standard head dims)?

3. **QMoE block-wise support:** Issue #29035 (closed) suggests ORT may have already added block-wise quantized expert weights. Should we verify before proposing our parallel routing as an improvement?

4. **Contribution strategy:** Should we propose these as individual PRs or a coordinated series? ORT's contrib process may prefer atomic, reviewable units.

5. **Licensing/attribution:** Our kernels are NVRTC-JIT (PTX from source strings). ORT uses compiled `.cu`. Is there any IP concern in porting the algorithmic approach?

6. **Priority vs. EP-compatibility milestone:** When is the earliest we could start a proof-of-concept PR for Candidate #1 (MatMulNBits block-128) without disrupting current work?

7. **DeepSeek MLA upstream demand:** Issue #23925 requests MLA on CPU/NPU. If upstream wants CUDA MLA, our CSA could potentially be contributed — but would require decoupling from `pkg.nxrt`. Worth pursuing?

---

## Methodology Notes

- **Verified:** Upstream file existence via GitHub code search API. Issue/PR status via GitHub issues search.
- **Verified:** Our kernel implementations via direct source reading.
- **Verified:** Performance claims from recorded benchmarks in `docs/benchmarks/` (H200, 2026-07-23 through 2026-07-27).
- **Inferred:** ORT's GQA decode quality (no source reading of their kernel internals).
- **Inferred:** ORT's QMoE block-wise support status (issue #29035 closed, may or may not be shipped).
- **Unverified:** Whether ORT's flash attention integration already includes split-KV decode.
- **Unverified:** Detailed comparison of our GEMV performance vs ORT's `fpA_intB_gemv` at matching shapes.
