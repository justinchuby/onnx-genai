### 2026-08-13: Graph-side glue node-collapse REALIZED — bf16 SiLU/SwiGLU-mul collapse ships (+0.9%, byte-exact)
**By:** Batty
**What:** Converted the §8 "+5.3% ceiling" GO (#899) into a realized, measured number on the
production Muse-Glimmer-30B native CUDA decode graph. The single graph-side change that is
byte-exactly collapsible on this model with a landed kernel: extend `CudaSiluFusion`
(`crates/onnx-runtime-ep-cuda/src/optimizer.rs`) to accept **BFloat16** (was `Float16`-only). The
standalone `Sigmoid(x)` + `Mul(x, sigmoid)` + `Mul(silu, up)` glue then collapses through
`CudaSiluFusion` → `CudaSwiGluFusion` into the tagged decomposed `Mul[_cuda_silu_mul]`, which the
runtime lowers to the already-landed **`decomposed_silu_mul_bf16`** epilogue (#867). Deletes 104
standalone nodes (2/layer × 52): `Sigmoid` 104→52, `Mul` 210→158, total decoder 2458→2354.
`CudaGateUpSwiGluFusion` needs an fp16 activation so it stays dormant → **int4 GEMVs untouched**
(per §7/#898). Two unit tests added (`collapses_bf16_decomposed_swiglu_glue`,
`leaves_fp32_decomposed_swiglu_glue_separate`).

**Measured (H200, `CUDA_VISIBLE_DEVICES=0`, `ONNX_GENAI_CUDA_GRAPH=1`, `--steady --warmups 1
--runs 3 --tokens 128`, 3 interleaved A/B rounds, same release binary):**
- decode: 21.19 → 20.99 ms/token; **throughput 47.20 → 47.63 tok/s = +0.9%**.
- node count/layer: ~22 glue → ~20 (−2/layer; −104 total).
- greedy 24-token stream **byte/token-identical** before↔after
  (`[24, 372, 1045, 10016, …, 1740, 2885]`).
- CUDA EP cdylib (`libonnx_runtime_ep_cuda_plugin.so`) built; `ldd` confirms it does **not** link
  `libonnxruntime`. `cargo fmt --check` + `cargo clippy` clean; 59 optimizer unit tests pass.

**Why:** On the real bf16 graph the elementwise/norm fusions were **dormant** because every
SiLU/SwiGLU pass was gated to `Float16`; Muse-Glimmer's stream is bf16. This change simply
activates the byte-exact bf16 kernel Sebastian already landed. Zero-risk: no cooperative kernel, no
`grid.sync`, no reduction reorder → no Chew gate.

**Verdict: SHIP** (small-but-real, byte-exact). Honest bound vs the +5.3% ceiling: only the 2
SiLU/SwiGLU-mul nodes/layer are byte-exactly collapsible here. The larger glue is blocked —
(1) **6 norms/layer**: Gemma3 sandwich-norm residual is `x + norm(y)`; #854's
`SkipSimplifiedLayerNormalization` (`norm(x+skip)`) *does* apply across the layer seam, **but only
f32/f16 skip kernels exist** — the f16 kernel rounds the residual sum to f16 before the RMS
reduction, whereas the f32-template path bf16 would reach accumulates over the **unrounded** fp32
sum → not byte-exact → **FLAGGED FOR CHEW/Sebastian: needs a bf16 skip-RMSNorm kernel that rounds
the sum**, then #854's fold can fire byte-exactly and recover the residual-Add + norm glue (the
bigger lever). (2) **~208 constant `gamma+1` Adds/layer**: #872 `CudaFoldConstantAdd` already
MEASURED a −2.8% regression — do not re-attempt. (3) **4 reshapes/layer**: GQA head-split metadata
coupled to the attention kernel, not free-standing deletions. Remaining glue interleaves with the
dominant GEMV serial cost, so realized ≤ ceiling as §8.3 predicted.

Doc: `docs/research/dense-decode-megakernel-feasibility.md` §8.5 (realized subsection). PR
references #899. Not self-merged.
