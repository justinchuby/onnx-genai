# Kimi K3 native-runtime readiness

**Audit point:** `main` at `c86cebc`, 2026-07-18

**Release boundary:** Kimi K3 is available through Moonshot's API, but weights and
the detailed technical report are promised by 2026-07-27. Exact tensor layouts,
checkpoint packing, layer schedule, and cache ABI are therefore not yet public
([official K3 announcement](https://www.kimi.com/blog/kimi-k3),
[official K3 API guide](https://platform.kimi.ai/docs/guide/kimi-k3-quickstart)).

> **Owner direction (2026-07-18):** defer K3-specific implementation and ABI
> decisions until official artifacts are released. The sections below remain a
> post-artifact readiness plan; model-agnostic infrastructure may proceed only
> when independently justified.

This document distinguishes:

- **verified K3 facts** from Moonshot;
- **lineage evidence** from Kimi K2, Kimi Linear, FlashKDA, and AttnRes; and
- **runtime facts** verified directly against the current CPU/CUDA registries.

## What changed since the previous audit

The prior Kimi audit was mostly right about the architectural gaps, but its
runtime snapshot is stale in two important ways:

1. CUDA now has **GPU-native** standard `ai.onnx::Attention` and
   `ai.onnx::RotaryEmbedding`, registered at opset 23/24 and opset 23
   respectively
   ([CUDA registry](../crates/onnx-runtime-ep-cuda/src/kernels/mod.rs#L416-L443)).
   Bulk Q/K/V, masks, outputs, and RoPE tensors remain device-resident
   ([standard attention](../crates/onnx-runtime-ep-cuda/src/kernels/standard_attention.rs#L27-L46),
   [RoPE kernel](../crates/onnx-runtime-ep-cuda/src/kernels/rotary_embedding.rs#L43-L65)).
   These landed in commits `7cefae9` and `4c9374b`, with correctness fixes in
   `53ef68c` and `74a891b`. Any older statement that these kernels are
   host-staged is superseded.
2. CPU now registers the complete frozen
   `pkg.nxrt::CompressedSparseAttention` v1 reference plus
   `pkg.nxrt::SparseKvGather`; CUDA registers neither on this `main`
   ([CPU registry](../crates/onnx-runtime-ep-cpu/src/kernels/mod.rs#L220-L231),
   [CUDA covered ops](../crates/onnx-runtime-ep-cuda/src/kernels/mod.rs#L94-L178)).
   A CUDA `SparseKvGather` port is in progress outside this audited tree; it must
   not be counted as landed. CUDA CSA has not started.

The CPU CSA implementation is substantial—ratio-4 and ratio-128 state, carries,
compressed records, FP8/FP4 cache formats, learned top-k, and shape inference
exist
([CSA header](../crates/onnx-runtime-ep-cpu/src/kernels/compressed_sparse_attention.rs#L1-L8),
[CSA factory](../crates/onnx-runtime-ep-cpu/src/kernels/compressed_sparse_attention.rs#L149-L263),
[shape registration](../crates/onnx-runtime-shape-inference/src/handlers/custom_ops.rs#L296-L313)).
It remains **DeepSeek CSA**, not Kimi KDA or MLA.

## Kimi K3 architecture summary

| Component | Current public evidence | Confidence / runtime implication |
|---|---|---|
| Scale and modality | 2.8T parameters, native vision, 1M-token context ([Moonshot](https://www.kimi.com/blog/kimi-k3)) | **High.** Model residency, expert distribution, and non-token visual inputs are first-order runtime concerns. |
| Attention | Kimi Delta Attention (KDA), Attention Residuals (AttnRes), and Gated MLA are named by Moonshot. The launch diagram appears to show a repeated 3× KDA / 1× Gated-MLA pattern, but no machine-readable layer schedule is published ([Moonshot](https://www.kimi.com/blog/kimi-k3)). | **High** for the named mechanisms; **medium** for the 3:1 schedule. Do not freeze layer counts from the diagram. |
| KDA state | FlashKDA exposes `q,k,v,g,beta`, gate parameters, and an initial/final recurrent matrix state `[B,H,V,K]`; its current kernel requires `K=V=128` ([FlashKDA](https://github.com/MoonshotAI/FlashKDA)). Kimi Linear additionally uses short convolution and a 3:1 KDA/global-MLA hybrid ([Kimi Linear](https://github.com/MoonshotAI/Kimi-Linear), [config](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Base/raw/main/config.json)). | **Medium as a K3 proxy, not a K3 ABI.** A closed vLLM FlashKDA integration documents an old/new gate-equation incompatibility, proving that “KDA” alone is not enough to select a kernel ([vLLM #43833](https://github.com/vllm-project/vllm/pull/43833)). |
| AttnRes | AttnRes attends over earlier layer/block representations; Block AttnRes reduces retained depth states to block representatives ([official AttnRes repository](https://github.com/MoonshotAI/Attention-Residuals)). | **High** for the general mechanism; K3 block count, projection layout, and exact placement remain unknown. This is activation state across depth, not token KV state. |
| MoE | Stable LatentMoE with 896 experts and 16 active experts per token. Quantile Balancing is named as the load-balancing method ([Moonshot](https://www.kimi.com/blog/kimi-k3)). | **High** for 896/top-16; **low** for the inference router ABI. The announcement does not establish whether quantile balancing changes inference-time selection or only training. Shared-expert count is not published. |
| Quantization | Quantization-aware training from SFT onward with MXFP4 weights and MXFP8 activations ([Moonshot](https://www.kimi.com/blog/kimi-k3)). | **High** for numeric families; **unknown** packing, scale granularity, block axes, accumulation dtype, and checkpoint encoding until release. |
| RoPE | No K3 RoPE configuration is public. K2 uses split non-RoPE/RoPE MLA dimensions plus YaRN ([K2 config](https://huggingface.co/moonshotai/Kimi-K2-Base/raw/main/config.json)); Kimi Linear's global MLA uses a 64-dimensional RoPE component and `rope_theta=10000` without scaling ([Kimi Linear config](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Base/raw/main/config.json)). | **Unknown for K3.** Standard partial-dimension RoPE is likely reusable for Gated MLA if the exporter supplies exact cos/sin tables, but the K3 frequency law must not be guessed. |
| MTP | Moonshot's K3 announcement and API guide do not identify an MTP/speculative head. K2 and Kimi Linear both publish `num_nextn_predict_layers=0` ([K2 config](https://huggingface.co/moonshotai/Kimi-K2-Base/raw/main/config.json), [Kimi Linear config](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Base/raw/main/config.json)). | **No verified K3 requirement.** Keep MTP conditional; “always-on reasoning” is an API behavior, not evidence of an MTP graph. |
| Deployment | Moonshot recommends supernodes with 64 or more accelerators and says KDA prefix-cache support is being contributed to vLLM ([Moonshot](https://www.kimi.com/blog/kimi-k3)). | **High.** Single-device correctness is useful, but practical K3 serving requires expert parallelism, collectives, and KDA-aware prefix state. |

## Capability coverage: what is reusable and what is genuinely missing

`✅` means the semantics are present. `◐` means useful pieces or a correctness
fallback exist. `❌` means a new semantic/runtime boundary is required.

| K3 need | Existing building block | Coverage | Exact assessment |
|---|---|---:|---|
| Ordinary dense/GQA attention | CPU and CUDA standard `Attention`; CPU/CUDA contrib `GroupQueryAttention` | ✅ | Recent CUDA standard Attention is GPU-native, but f32-only and uses conventional full K/V caches ([kernel constraints](../crates/onnx-runtime-ep-cuda/src/kernels/standard_attention.rs#L69-L75)). It is a correctness fallback for reconstructed K/V, not KDA or efficient MLA. |
| RoPE for a global-attention fallback | CPU and CUDA `RotaryEmbedding` | ◐ | Both support partial rotary dimensions and precomputed cos/sin tables. CUDA is GPU-native but f32-only. Reuse is likely once K3's exact RoPE tables/interleaving are known; the frequency law is not known today. |
| Efficient MLA latent cache | `attention.type: multi_latent` metadata vocabulary; MatMul/RoPE/Attention decomposition | ❌ | **There is no native MLA operator or MLA cache adapter.** The metadata value is descriptive only ([schema](../crates/onnx-genai-metadata/src/schema.rs#L225-L263)). Expanded K/V through standard Attention is possible but forfeits MLA's cache benefit. |
| Gated MLA | Standard linear, normalization, RoPE, Attention pieces | ❌ | Gate math may decompose into graph ops, but learned low-rank latent state, decoupled RoPE cache, and gate-bearing fused execution are absent. Exact K3 semantics are unpublished. |
| KDA recurrent attention | CPU CSA state machinery; generic graph I/O | ❌ | CSA is temporal compression plus sparse selection at fixed ratios 4/128. KDA is a gated delta-rule recurrent matrix update with different equations and state. Reuse CSA's versioned-op, shape, state/checkpoint, and golden-test patterns—not its operator or kernel. |
| KDA prefix cache / rollback | Paged KV checkpoints; CSA compressed-state/carry precedent | ❌ | Current engine state is token K/V-centric. KDA needs typed recurrent and convolution state, prefix import/composition rules, capacity accounting, and speculative rollback. Moonshot explicitly says conventional prefix caching needs a KDA-specific implementation. |
| AttnRes | MatMul/RMSNorm/Softmax and generic SSA values | ◐ | A portable decomposition is plausible. New work is activation-lifetime planning and retaining block representations across depth; a fused op is optional until profiling. Do not map AttnRes to KV pages. |
| 896-expert, top-16 sparse routing | CPU `MoE`/`QMoE`; CUDA `QMoE` grouping/GEMM | ◐ | Dynamic expert count/top-k and selected-route grouping are reusable. Stable LatentMoE's router and any inference-visible quantile rule must be verified. Current QMoE accepts only affine integer `quant_type="int"` ([CPU](../crates/onnx-runtime-ep-cpu/src/kernels/qmoe.rs#L79-L89), [CUDA](../crates/onnx-runtime-ep-cuda/src/kernels/qmoe.rs#L709-L723)). |
| MXFP4 dense weights | CPU/CUDA `pkg.nxrt::BlockQuantizedMatMul` | ◐ | Both backends decode the repository's OCP/llama-style MXFP4 blocks. This is reusable only after byte-level proof that Moonshot's checkpoint packing matches, or after an explicit conversion. CUDA currently consumes f32 activations/outputs, not MXFP8. |
| MXFP4/MXFP8 routed experts | Proposed `pkg.nxrt::BlockQuantizedMoE`; QMoE route grouping; block decoders | ❌ | `BlockQuantizedMoE` is design-only and unregistered. The lazy-weight seam recognizes its name but can bind/materialize only whole weights, not selected expert slices ([design](BLOCKQUANTIZEDMOE_DESIGN.md), [weight seam](../crates/onnx-runtime-ep-api/src/weight.rs#L94-L105), [binder](../crates/onnx-runtime-ep-api/src/weight.rs#L205-L243)). K3 also needs an exact MXFP8 activation contract. |
| Expert weight paging | External mmap, `WeightHandle`, lazy-boundary negotiation | ◐ | Host materialization fallback exists; live device paging and per-expert leases do not. A 2.8T model cannot treat whole-weight materialization as the production path. |
| Multi-device expert parallelism | `MOE_EXPERT_PARALLELISM.md` design | ❌ | No `MoeDispatch`/`MoeGather`, NCCL all-to-all, per-node placement, or distributed KDA/MLA state is implemented. This is required for practical deployment, independently of single-device kernel correctness. |
| MTP/speculation, if K3 publishes it | Generic speculative loop and `MtpProposer` | ◐ / conditional | The reusable draft/verify/accept loop exists, but metadata still marks MTP unsupported and the proposer is rebuilt each iteration ([parser](../crates/onnx-genai-metadata/src/parser.rs#L70-L90), [loop](../crates/onnx-genai-engine/src/speculative.rs#L965-L980)). Do not make K3 readiness depend on this without an artifact. |
| Native vision front end | Existing generic ONNX graph/runtime arithmetic | ❌ / unknown | K3 verifies native image/video understanding, but no encoder, projector, tiling, positional, or package contract is public. Audit the released graph instead of guessing from Kimi-VL. |

### Building-block verdict

- **CSA / `CompressedSparseAttention`: reuse the engineering pattern, not the
  semantics.** It proves the CPU EP can own versioned compressed state, carries,
  sparse selection, quantized cache records, and shape inference. It does not
  implement KDA or MLA.
- **MLA: not currently implemented as a runtime building block.** We have a
  metadata label and decomposable projections/Attention/RoPE, but no latent-cache
  operator or lifecycle.
- **MoE/QMoE: routing/grouping is reusable; K3 quantization is not covered.**
  `BlockQuantizedMatMul` supplies useful MXFP4 decode math, while
  `BlockQuantizedMoE` and MXFP8 activation execution remain missing.
- **MTP: engine scaffolding exists but is incomplete and not a verified K3
  requirement.**
- **RoPE and standard Attention: recently landed GPU-native and are valid
  fallback/oracle primitives.** Their f32 conventional-KV contract is not a
  production KDA/Gated-MLA implementation.

## Prioritized gaps

### P0 — freeze the released K3 contract immediately

On weight/report release, capture and golden-test:

1. exact KDA gate equation, convolution state, recurrent-state shape/dtype/layout,
   prefill/decode transition, and prefix-cache transform;
2. Gated MLA projection/gate equations, latent cache, RoPE split/scaling, and
   layer schedule;
3. Stable LatentMoE router scoring, top-16 normalization/tie behavior, shared
   experts, and whether Quantile Balancing is inference-visible;
4. MXFP4/MXFP8 byte layout, scale granularity, accumulator dtype, and expert
   tensor axes; and
5. vision graph/package inputs plus any MTP sidecar.

Do not freeze a private ABI solely from Kimi Linear: vLLM #43833 already shows
that two KDA generations can use incompatible gate equations.

### P0 — add typed KDA state and CPU/CUDA KDA kernels — genuinely new

Add a versioned semantic operator (for example
`pkg.nxrt::KimiDeltaAttention`) with explicit Q/K/V/gate/beta inputs,
initial/final recurrent state, convolution state, variable-length batching, and
prefill/decode modes. Extend the runtime state manager beyond `[K,V]` pages to
typed attention state with checkpoint/restore/fork/prefix-import. Implement a
scalar/CPU oracle first, then CUDA; FlashKDA is a backend candidate only when
the released gate/layout contract matches.

### P0 — add native Gated MLA latent-cache execution — genuinely new

Create a model-agnostic MLA boundary with explicit low-rank latent state,
non-RoPE/RoPE dimensions, gate inputs, and past/present latent outputs. Reuse
current MatMul/BlockQuantizedMatMul, RoPE, and standard Attention for a
decomposed oracle, but do not ship expanded K/V as the production 1M-context
path.

### P0 — implement model-exact block-quantized MoE and expert leases

Reuse QMoE routing/grouping and BlockQuantizedMatMul's decode infrastructure,
but implement and register `pkg.nxrt::BlockQuantizedMoE` on CPU/CUDA, add exact
Moonshot MXFP4 packing or an explicit converter, support the required MXFP8
activation path, and extend lazy binding to selected expert slices. This is
partly reuse, but the op, activation format, and lease granularity are new.

### P0 for deployment — implement expert parallelism

The single-EP executor cannot practically host 2.8T parameters. Implement
per-node/per-region placement, expert ownership, GPU-native dispatch/combine
collectives, shared-expert policy, and distributed attention state. The existing
session-per-GPU document is a design baseline, not implementation evidence
([MOE_EXPERT_PARALLELISM.md](MOE_EXPERT_PARALLELISM.md)).

### P1 — AttnRes activation residency and optional fusion

First export a portable Block-AttnRes decomposition and teach liveness planning
to retain only required block representatives. Add a fused op only if profiling
shows material launch/memory cost.

### P1 — production dtypes and fallback quality

Add f16/bf16—and, where the released contract requires it, MXFP8—support to
CUDA Attention/RoPE/normalization paths. The current GPU-native f32 kernels are
excellent correctness oracles but are not sufficient evidence for efficient
K3 precision.

### Conditional — MTP and vision

- Do not prioritize MTP until the released package contains a draft head or
  sidecar. If it does, reuse the speculative state machine but fix package
  discovery and persistent proposer/state rollback.
- Audit the released vision encoder/projector graph as a separate P0/P1 intake.
  Native vision is verified, but its runtime contract is not public.

## Deferred decisions for Justin

1. **Pre-build before 2026-07-27?** Recommended: defer K3-specific
   implementation until official artifacts arrive. Continue only
   model-agnostic infrastructure that is independently justified; do not add a
   provisional K3 operator ABI.
2. **Quantization policy?** Choose whether native K3 packages must preserve
   Moonshot MXFP4/MXFP8 exactly, or whether an explicit conversion profile is an
   acceptable first milestone. Recommended: defer this choice until official
   packing and activation semantics are available. Do not label an existing
   generic `mxfp4` decoder as K3/Moonshot compatible.
3. **First deployment target?** Choose single-GPU/small-shard correctness versus
   immediate 64+-accelerator production work. Recommended: defer the K3-specific
   deployment target until official artifacts establish the runnable layout and
   resource profile. Model-agnostic expert-parallel transport/placement may
   proceed independently.
4. **MLA/KDA boundary strategy?** Recommended: preserve the guardrail that
   `KDA`, `MLA`, and `CSA` are separate semantic state kinds, but defer
   K3-specific KDA/MLA ABI and implementation until official artifacts arrive.
5. **MTP scope?** Recommended: defer all K3-specific MTP work until
   weights/config verify a draft head or sidecar.

## Bottom line

The runtime is better positioned than the previous audit implied: CPU CSA is a
real stateful reference implementation, and standard CUDA Attention plus RoPE
are now GPU-native. Those are valuable oracle and infrastructure pieces.

They do **not** close the K3-critical gaps. The missing production semantics are
KDA recurrent state/prefix caching, native Gated MLA latent caching,
model-exact MXFP4/MXFP8 MoE with selected-expert leases, AttnRes activation
residency, and multi-device expert parallelism. MTP is unverified and should not
drive the schedule before the 2026-07-27 artifact release.
