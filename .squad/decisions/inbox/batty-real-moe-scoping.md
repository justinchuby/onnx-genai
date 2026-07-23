### 2026-07-23: Use DeepSeek-V2-Lite as the first real native-CUDA MoE target
**By:** Batty
**What:** Target DeepSeek-V2-Lite first. Its full model is plausibly single-H200 runnable, but production-quality execution requires (1) an MLA-aware attention/cache lowering and (2) a real expert-major int4 `QMoE` emitter/packer. The CUDA `QMoE` kernel has no fixed expert-count or top-k ceiling; the missing exporter and real-scale validation are the immediate MoE gaps.
**Why:** Qwen1.5-MoE is slightly smaller and is a useful lower-risk stepping stone, but DeepSeek-V2-Lite is the smallest candidate that exercises the breadth target: low-rank MLA with decoupled RoPE, 64 fine-grained routed experts, top-6 routing, and shared experts.

# Real production MoE native-CUDA scoping

## Decision

**Primary target: `deepseek-ai/DeepSeek-V2-Lite` (16B total, 2.4B active).**

GPU4 is an H200 with 143,771 MiB total memory, so either bf16 or int4 structural
work is feasible on one device. DeepSeek-V2-Lite is not the smallest candidate
by total parameters, but it is the best breadth target that is still practical:

| Candidate | Size | Attention | Routed MoE | Export status |
|---|---:|---|---|---|
| Qwen1.5-MoE-A2.7B | 14.3B total / 2.7B active | MHA (16 Q / 16 KV heads) | 60 experts, top-4, plus a 5632-wide shared expert | Mobius registers `qwen2_moe`; ORT GenAI 0.14.1 builder has no `Qwen2MoeForCausalLM` dispatch. Mobius currently statically unrolls experts. |
| **DeepSeek-V2-Lite** | **16B total / 2.4B active** | **MLA:** 16 heads, 512-wide KV latent, 128 non-RoPE + 64 RoPE Q/K dimensions, V dimension 128 | **64 routed + 2 shared, top-6**; softmax/greedy routing, scale 1.0 | Mobius registers `deepseek_v2`; ORT GenAI builder has no DeepSeek dispatch. Current Mobius graph is structurally exportable but statically unrolls experts. |
| Qwen3-30B-A3B | 30.5B total / 3.3B active | GQA (32 Q / 4 KV heads) | 128 experts, top-8 | Mobius registers `qwen3_moe`; ORT GenAI builder has no `Qwen3MoeForCausalLM` dispatch. |
| GLM-4.5-Air | 106B total / 12B active | partial-RoPE GQA (96 Q / 8 KV heads) | 128 routed + 1 shared, top-8 | Mobius registers `glm4_moe`; ORT GenAI builder has no `Glm4MoeForCausalLM` dispatch. Much larger than the first target. |
| GLM-5 | 744B total / 40B active | DSA, 64 Q / 64 KV heads; MLA-like 512 KV and 2048 Q low-rank paths | 256 routed + 1 shared, top-8 | Tiny branch exporter exists, but current Mobius registry has no real `glm_moe_dsa` registration. Not a single-device first target. |

Architecture evidence comes from the official Hugging Face `config.json` and
model cards:

- <https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B>
- <https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite>
- <https://huggingface.co/Qwen/Qwen3-30B-A3B>
- <https://huggingface.co/zai-org/GLM-4.5-Air>
- <https://huggingface.co/zai-org/GLM-5>

Builder evidence:

- ORT GenAI 0.14.1 dispatches dense Qwen2/Qwen3 and Qwen3.5-MoE, but not the
  candidate architectures:
  `/home/justinchu/.local/lib/python3.12/site-packages/onnxruntime_genai/models/builder.py:215-332`.
- Mobius registrations:
  `/home/justinchu/mobius/src/mobius/_registry.py:520-544,885-910`.
- Mobius's generic routed MoE is explicitly a loop over every expert:
  `/home/justinchu/mobius/src/mobius/components/_moe.py:183-228`.

Qwen1.5-MoE is the recommended **secondary fixture** after DeepSeek export
plumbing exists: it isolates real-scale QMoE from MLA, but choosing it as the
primary target would postpone the most important architectural breadth gap.

## Attention gap: correctness can decompose, production MLA cannot

There is **no native MLA operator or latent-cache adapter** in this repository.
The existing native op vocabulary has standard `Attention`, GQA, DSA/CSA, and
IndexShare, but no `MLA`, `kv_lora`, `q_lora`, or compressed-MLA cache boundary.

Mobius does implement the DeepSeek equations as primitive graph operations:

- low-rank Q and KV projections;
- split latent KV and decoupled RoPE key;
- decompress to per-head K-nope and V;
- concatenate the RoPE/non-RoPE portions;
- call standard `ai.onnx::Attention`.

See `/home/justinchu/mobius/src/mobius/components/_deepseek_mla.py:29-218`.
Thus MLA is **not a new op for a first correctness run**. It is a new runtime
boundary for production memory, capture, and throughput.

### Smoke result

A weightless real-config build succeeded:

```bash
/home/justinchu/mobius/.venv/bin/mobius build \
  --model deepseek-ai/DeepSeek-V2-Lite \
  --no-weights --dtype f16 --ep default \
  <output-dir>
```

The graph had 26,226 nodes, including 27 standard `Attention` nodes and 5,208
`MatMul` nodes. It exposed expanded cache tensors:

```text
past_key_values.N.key   [batch, 16, past_sequence_len, 192]
past_key_values.N.value [batch, 16, past_sequence_len, 128]
```

The native standard-Attention CUDA kernel supports distinct Q/K and V head
sizes (`v_head_size`) at
`crates/onnx-runtime-ep-cuda/src/kernels/standard_attention.rs:823-924`.
However, it synchronizes before/after execution and explicitly declines CUDA
graph capture:
`standard_attention.rs:713-717,1125,1142-1145`.

There is also a concrete exporter bug for `--ep cuda`: Mobius rewrites all 27
standard Attention nodes to `com.microsoft::GroupQueryAttention`. Its rewrite
does not guard unequal K and V head dimensions
(`/home/justinchu/mobius/src/mobius/rewrite_rules/_group_query_attention.py:122-150,213-238`).
Native GQA requires V to have the same hidden width as K and rejects this graph
(`crates/onnx-runtime-ep-cuda/src/kernels/group_query_attention.rs:1368-1397`).
The first export must therefore retain standard Attention, or the rewrite must
decline MLA shapes.

Expanded MLA cache costs 5,120 bf16 elements/layer/token versus approximately
576 elements for the KV latent plus RoPE key: about **8.9x**. Across 27 layers,
the expanded cache is about **8.44 GiB at 32K tokens** (about 42.2 GiB at the
configured 163,840 maximum), versus about 0.95 GiB (4.75 GiB) for latent state.
It fits H200, but it is not production MLA.

**Conclusion:** standard Attention is a valid correctness fallback. The #1
production architecture blocker is a capture-safe MLA boundary that keeps the
compressed latent/decoupled-RoPE cache rather than materializing full K/V.

## MoE/QMoE gap

The tiny fixture uses 4 routed experts, top-2, fused-SwiGLU int4 block-32
`QMoE`. Its node has expert-major packed FC1/FC2 tensors and separate selection
and aggregation inputs.

DeepSeek-V2-Lite needs:

- 64 routed experts, top-6;
- FC1 gate/up and FC2 down expert-major int4 packing;
- 2 always-on shared experts;
- explicit softmax/greedy routing (V2-Lite has no group restriction);
- `routed_scaling_factor=1.0`.

The CUDA kernel has **no hardcoded 4-expert/top-2 limit**:

- it accepts any `0 < k <= num_experts`:
  `crates/onnx-runtime-ep-cuda/src/kernels/qmoe.rs:983-995`;
- route, scratch, grouping, and GEMV sizes are checked runtime products:
  `qmoe.rs:1111-1145,1320-1365,1568-1635`;
- grouping loops over runtime expert counts:
  `qmoe_grouping.rs:19-105`;
- grouped GEMM indexes runtime expert-major tensors:
  `qmoe_gemm.rs:64-147`.

The route kernel is a serial top-k scan per token (`experts × top_k`, with a
small duplicate-selection loop) at `qmoe.rs:70-95`. The grouping prefix is a
single-thread scan over experts (`qmoe_grouping.rs:66-81`). These are not
correctness limits for 64/top-6, but real-scale performance is unmeasured.
Current GPU tests cover at most 4 experts/top-2
(`crates/onnx-runtime-ep-cuda/tests/qmoe_gpu.rs:604-931`).

Shared experts should remain outside QMoE as an always-on dense gated MLP and
be added to routed output. Mobius already models this correctly:
`/home/justinchu/mobius/src/mobius/models/deepseek.py:240-277`.
QMoE input 14 (`router_weights`) supports separate selection and aggregation,
including DeepSeek-style routing and external scaling:
`qmoe.rs:56-133,1081-1099`. V2-Lite's simpler softmax/greedy, scale-1 routing
does not require a kernel ABI extension.

**Conclusion:** the kernel ABI generalizes. The blocking gap is the exporter:
Mobius splits 3-D checkpoint expert tensors into thousands of per-expert
MatMuls (`deepseek.py:374-415`) and does not emit packed QMoE. The smoke graph
contained 5,208 MatMuls, zero QMoE nodes, and 26,226 total nodes. A production
artifact needs a fused QMoE emitter/packer and a 64/top-6 differential test.

## Generation and ORT comparison status

### What works now

Mobius can produce a float structural graph from the real checkpoint/config:

```bash
/home/justinchu/mobius/.venv/bin/mobius build \
  --model deepseek-ai/DeepSeek-V2-Lite \
  --dtype f16 --ep default --runtime onnx-genai \
  /home/justinchu/glm-e2e-artifacts/deepseek-v2-lite-f16
```

Use `--ep default` until the unequal-K/V GQA rewrite is guarded. This command
would download/build the full checkpoint, so only `--no-weights` was run during
this scoping session.

### What does not work now

There is no current one-command production int4 export:

- ORT GenAI 0.14.1's builder has no `DeepseekV2ForCausalLM` dispatch.
- Mobius does not quantize a bf16 checkpoint. It can preserve supported
  pre-quantized linear weights, but the DeepSeek path currently explodes the
  routed tensors into per-expert modules rather than packing one QMoE node.
- Therefore the nominal future command

  ```bash
  /home/justinchu/mobius/.venv/bin/mobius build \
    --config <local-supported-int4-DeepSeek-V2-Lite-checkpoint> \
    --dtype f16 --ep cuda --runtime onnx-genai \
    <output-dir>
  ```

  is blocked on the QMoE emitter/packer and MLA GQA-fusion guard.

ORT GenAI 0.14.1 load/comparison status is **not yet established**, rather than
a demonstrated “ORT cannot load” result: no correctly packed real artifact
exists to load. Its bundled ORT 1.27 runtime contains standard Attention and
the current QMoE schema (including DeepSeek-style separate router weights), so
an ORT load smoke must be the acceptance test immediately after artifact
generation. The ORT builder itself cannot be the producer.

## Prioritized enablement plan

1. **P0 — make the real MLA graph structurally runnable (2-4 days).**
   Guard Mobius GQA fusion when K/V head dimensions differ; retain standard
   Attention for the first artifact. Add a no-weight DeepSeek-V2-Lite graph
   contract test asserting 27 Attention nodes, 192-dimensional K, 128-dimensional
   V, and no GQA. Then run one-layer/full-graph fp16 parity against HF/ORT.
   This separates attention correctness from MoE packing.

2. **P0 — implement real expert-major int4 QMoE emission/packing (1-2 weeks).**
   Consume DeepSeek fused 3-D gate/up/down tensors (and supported GPTQ/AWQ
   metadata), pack FC1/FC2/scales/zero-points exactly to the QMoE ABI, preserve
   the explicit router subgraph, feed separate aggregation weights when needed,
   and leave the two shared experts as a dense int4 MLP. Differentially compare
   fused QMoE against Mobius's static 64-expert reference.

3. **P0/P1 — validate QMoE at real routing scale (2-4 days).**
   Add synthetic CUDA/CPU parity for 64 experts/top-6, empty/hot experts,
   prefill grouping, fp16/bf16, capture warm/replay, and memory sizing. Profile
   the serial route/prefix stages; optimize only if material at 64/top-6.

4. **P1 — generate and smoke the complete int4 package (3-5 days plus download/build time).**
   Produce the Mobius artifact, inspect that each MoE layer has one QMoE plus
   one shared dense MLP, run 1-8 greedy tokens on GPU4 with zero fallbacks, and
   compare logits/tokens with HF. Attempt ORT GenAI 0.14.1 load and generation
   on the identical package; record either throughput or the exact loader/schema
   failure.

5. **P1 — add production MLA latent-cache execution (2-4 weeks).**
   Define a model-agnostic MLA op/state contract for compressed KV plus
   decoupled RoPE, implement CPU oracle and capture-safe CUDA execution, and
   teach metadata/engine KV handling the latent state. This is the critical
   path from “runs on H200” to production MLA memory and throughput.

6. **P2 — performance and breadth ladder (1-2 weeks, then ongoing).**
   Tune grouped expert GEMM and routing, then validate Qwen1.5-MoE (real QMoE
   without MLA), Qwen3-30B-A3B (128/top-8 GQA), GLM-4.5-Air, and finally
   GLM-5/DSA+MTP. Do not jump to GLM-5 until DeepSeek-V2-Lite proves the packed
   real-MoE and MLA-state contracts.

**Critical path:** attention lowering guard → QMoE emitter/packer → 64/top-6
parity → full int4 artifact/ORT smoke → latent-cache MLA optimization.
