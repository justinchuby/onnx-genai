### 2026-08-13: bf16 SwiGLU fusion — kernel side done; two optimizer bf16 gates are the blocker (Batty); full gate_up fold non-deterministic (scoped follow-up)

**By:** Sebastian (Performance/Systems)

**What:**
Investigated the coordinator-assigned "SwiGLU Mul into the GEMV epilogue" deliverable. Key correction: **it is NOT self-contained in the kernel.** The fused gate/up SwiGLU kernel already exists and already folds `silu(gate)*up` into the epilogue (#867). Muse-Glimmer's standalone `Mul` survives purely because the graph-rewrite passes that emit the fused node are **bf16-gated** in `optimizer.rs` (Batty's file). My only real kernel gap was a missing bf16 *decomposed*-SiLU kernel, now shipped.

**Kernel side (DONE — my PR, branch `squad/cuda-kernel-epilogue-fusion`):**
- Added `decomposed_silu_mul_bf16` + `decomposed_silu_bf16` NVRTC kernels and admitted `BFloat16` in the decomposed path of `SiluMulKernel` and `UnaryKernel` (`kernels/elementwise.rs`). Previously bf16 hit a hard `"decomposed SiLU fusion requires float16"` error — a real portability defect.
- Byte-exact: Sigmoid and the silu product each round to bf16 via the same `__float2bfloat16_rn` the standalone `sigmoid_bf16`/`mul_bf16` ops use; all intermediate math fp32. Added an f64/bf16 oracle test asserting bit-equality vs the unfused two-op graph. fmt+clippy clean; 5/5 silu tests pass.
- The fused `gate_up_swiglu` (+rmsnorm, +zp, +decomposed) kernel is ALSO already bf16-ready via `MatMulNBitsKernel::run_bf16` staging (bf16→f16→kernel→bf16, dual-scale const-cache `cache_slots=[2,4]`). No kernel change needed there for correctness.

**Graph side — ASK for Batty (optimizer.rs, two bf16 gates):**
1. **`CudaSiluFusion` (optimizer.rs:129):** `if graph.value(x).dtype != DataType::Float16 { continue; }` — rejects the bf16 Sigmoid input, so the whole `x*Sigmoid(x) → Silu` rewrite never fires for Muse-Glimmer. Admit `BFloat16` alongside `Float16` (one-liner). **This alone** activates the SAFE fusion: `Sigmoid+Mul → silu_mul` (my new bf16 kernel), collapsing 2 nodes/layer. `CudaSiluMulFusion` already admits bf16 (optimizer.rs:1585).
2. **`CudaGateUpSwiGluFusion::eligible_projection` (optimizer.rs:~1829):** requires gate/up MatMulNBits activation/output/scales == `Float16`. Muse-Glimmer's are `BFloat16`, so the FULL fold (two GEMVs + silu → one node) is rejected and falls back to standalone silu_mul. **Do NOT relax this yet — see the non-determinism finding below.**

**Measured (H200, ONNX_GENAI_CUDA_GRAPH=1, --pipeline, staged Muse-Glimmer, capture 1seg/0seams throughout; I temporarily patched the gates locally to measure, then reverted — those patches are Batty's to land):**
- Baseline: **2568 nodes/token, 47.20 tok/s.**
- Gate #1 only (safe `silu_mul` fold + my bf16 kernel): **2464 nodes (−104), 47.64 tok/s (flat within noise), first-16 greedy ids byte-exact.** Confirms the earlier dispatch-bound finding: the removed Sigmoid/Mul are the *cheapest* nodes, so −4% nodes ≈ 0% tok/s.
- Gate #1 + #2 (full gate_up bf16 fold): **2256 nodes (−312)** BUT decode became **NON-DETERMINISTIC across runs** ("pipeline greedy decode was not deterministic"). The bf16→f16 *staging* path through the fused dual-scale gate_up kernel is not capture-stable/deterministic (standalone bf16 MatMulNBits staging IS deterministic, so it's specific to the fused dual-scale + larger-output path).

**Scoped follow-up (my domain, needs Chew):** making the full gate_up bf16 fold safe needs a **bf16-native fused `gate_up_swiglu` kernel** (bf16 in/out, fp32 accumulate — mirror `gqa_decode_bf16` / the #867 matmul bf16 path), removing the f16-staging round-trip that is the likely non-determinism source. That is a genuinely separate kernel-family effort (base/rmsnorm/zp/decomposed variants + oracle). It is the real 47→higher lever (−312 nodes incl. the *expensive* GEMV nodes), but it is not a quick win and must clear Chew's numerics gate. Flagging to the coordinator rather than grinding.

**Why:** The SwiGLU-fusion node-count lever is graph-gated on bf16, not kernel-gated. The safe half (gate #1 + my kernel) is byte-exact but perf-flat (dispatch-bound, cheapest nodes). The valuable half (full gate_up fold) is blocked on a bf16-native fused kernel to be capture-deterministic. Recommend: Batty lands gate #1 to compose with my kernel PR; the full fold is a scoped bf16-native-kernel follow-up (Sebastian + Chew), coordinator-directed.
