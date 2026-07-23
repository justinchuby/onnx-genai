# GLM / DeepSeek int4 CUDA enablement scope

Research date: 2026-07-23. This document scopes model/export work only; it does
not propose CUDA kernel changes.

## Inventory

The materialized Foundry cache contains no GLM, Gemma, or DeepSeek directory.
The relevant inventory is:

| Family | Materialized path | Format / evidence |
|---|---|---|
| DeepSeek | none | `find ~/.foundry/cache -iname '*deepseek*'` returned no directories. `foundry.modelinfo.json` advertises uncached CUDA and CPU DeepSeek-R1-Distill-Qwen 7B/14B entries, but every entry has `"cached": false`. |
| GLM | none | No GLM entry is materialized or advertised by the local manifest. |
| Gemma | none | No Gemma directory is materialized. |
| Qwen3 | none | The manifest advertises Qwen3 CUDA/CPU variants, all uncached. |
| Qwen3.5 | `/home/justinchu/.foundry/cache/models/Microsoft/qwen3.5-9b-generic-cpu-2/v2` | CPU-only materialization. `text.onnx` has 249 `com.microsoft::MatMulNBits` nodes but also unsupported native-CUDA ops `LinearAttention` and `CausalConvWithState`; this is not a usable CUDA-native decoder. |
| Existing CUDA controls | `.../qwen2.5-{0.5b,1.5b,7b}-instruct-cuda-gpu-4/` and `.../Phi-4-mini-instruct-cuda-gpu-5/` | Known CUDA packages used as format controls. |

Manifest: `/home/justinchu/.foundry/cache/models/foundry.modelinfo.json`.
The manifest advertises `deepseek-r1-distill-qwen-7b-cuda-gpu:4` (5,406 MB)
and 14B CUDA, but the `foundry` executable is not installed:

```text
/bin/bash: foundry: command not found
```

Therefore those catalog entries cannot be fetched through the local CLI in
this environment.

## Native CUDA graph contract

The required weight operator is `com.microsoft::MatMulNBits` v1:

- attributes: positive `K`, `N`; `bits=4`; standard non-prepacked layout;
- block size: any power of two >=16 is correct, with block-32 the tuned/common
  path (block-128 is also implemented);
- packed weight `B`: `uint8 [N, ceil(K/block_size), block_size*bits/8]`;
- scales: `[N, ceil(K/block_size)]`, fp16 for an fp16 graph (fp32 is accepted);
- zero points: optional packed uint8. Missing means symmetric midpoint 8;
  explicit packed zero points enable asymmetric uint4;
- fp16 activations/outputs are the preferred CUDA path. `accuracy_level` may be
  omitted/0; level 4 has a specialized symmetric block-32 path.

Evidence: `crates/onnx-runtime-ep-cuda/src/kernels/matmul_nbits.rs:2253-2433`
and Mobius's layout definition in
`../mobius/src/mobius/components/_quantized_linear.py:20-26,134-156`.

A dense GLM/DeepSeek decoder must restrict itself to CUDA-covered graph ops.
The transformer-critical set is:

- `MatMulNBits`, plus `MatMul`/`Gemm` for projections intentionally left float;
- `GroupQueryAttention` (`com.microsoft`, v1) **or** standard
  `ai.onnx::Attention` opset 23/24;
- standard `RotaryEmbedding` opset 23, or GQA's fused RoPE. Partial RoPE is
  supported through `rotary_embedding_dim`; interleaved and rotate-half forms
  are supported;
- `RMSNormalization` / `SimplifiedLayerNormalization` and
  `SkipSimplifiedLayerNormalization`;
- ordinary shape, movement, elementwise, reduction, `TopK`, `GatherElements`,
  and `OneHot` operators listed in
  `crates/onnx-runtime-ep-cuda/src/kernels/mod.rs:108-197`.

GQA requires positive `num_heads`/`kv_num_heads`, divisible head groups,
unquantized KV (`k_quant_type=v_quant_type=NONE`), and no `qk_output`,
`smooth_softmax`, or quantized KV cache. Partial fused RoPE is accepted from
the cos/sin cache width. See
`group_query_attention.rs:801-869,1469-1556`.

## Emitters and exact invocations

### ONNX Runtime GenAI builder: works now for dense Llama/Qwen/ChatGLM families

The repository venv contains `onnxruntime-genai 0.14.1`, but its builder is
broken because a runtime dependency is absent:

```text
$ .ort-genai-cuda-venv/bin/python -m onnxruntime_genai.models.builder --help
ModuleNotFoundError: No module named 'onnx_ir'
```

The user Python installation has a working builder. It supports
`LlamaForCausalLM`, `Qwen2ForCausalLM`, `Qwen3ForCausalLM`, and legacy
`ChatGLMForConditionalGeneration`, but has no DeepSeek-V2/V3 architecture
dispatch (`~/.local/lib/python3.12/site-packages/onnxruntime_genai/models/builder.py:216-328`).

The smallest native-DeepSeek-branded dense checkpoint was generated
successfully:

```bash
python3 -m onnxruntime_genai.models.builder \
  -m deepseek-ai/deepseek-coder-1.3b-base \
  -o /home/justinchu/glm-e2e-artifacts/deepseek-coder-1.3b-int4-cuda \
  -p int4 -e cuda \
  --extra_options int4_block_size=32 int4_is_symmetric=true \
                  int4_accuracy_level=0 hf_token=false
```

Output is 820 MB and contains 121 symmetric block-32 int4 `MatMulNBits`, 24
GQA, 48 skip-RMSNorm, and no uncovered graph op. A native CUDA smoke passed:

```bash
ONNX_GENAI_CUDA_KV_MAX_LEN=64 ONNX_GENAI_CUDA_GRAPH=0 \
cargo run -q -p onnx-genai-bench --features bench-native,cuda \
  --bin profile_native -- \
  --model /home/justinchu/glm-e2e-artifacts/deepseek-coder-1.3b-int4-cuda \
  --ep cuda --tokens 4 --warmups 0 --runs 1 --prompt 'def fibonacci(n):'
```

Result: 4 tokens, coherent continuation `"\n    if n"`, 1.01 tok/s, and
`fallbacks=0`. This proves the weight-production path and native CUDA graph
compatibility; performance tuning is explicitly out of scope here.

Artifact checksums:

```text
64cee2143814cb1992699debfe80c4bcaa130bc64f14e68462f7bc5b7ca98eaa  model.onnx
a9b17eabd4bc25d4c52ea1a39f328a24db785356501abb21aca5822433327236  model.onnx.data
f0e8cb2f611c87ce8b1232c992fff98ea79a5f61d1d459c77ac25fbe6e3aa7da  genai_config.json
```

The same builder can target DeepSeek-R1-Distill-Qwen-1.5B because its HF
architecture is `Qwen2ForCausalLM`, but that is a Qwen architecture rather than
a native DeepSeek architecture.

### Mobius

Mobius supports `glm`, `glm4`, `glm4_moe`, `deepseek_v2`, and `deepseek_v3`.
Its normal CLI does not quantize a float checkpoint: it emits `MatMulNBits`
when the input checkpoint already carries GPTQ/AWQ/Olive/GGUF quantization
metadata and packed tensors. Exact build form:

```bash
/home/justinchu/mobius/.venv/bin/mobius build \
  --model <quantized-HF-model-or-local-config> \
  --dtype f16 --ep cuda --runtime onnx-genai \
  --optimize=group_query_attention,skip_norm \
  <output-dir>
```

Branch `origin/glm5.2-moe-export` additionally has
`export_glm_tiny_quant.py` and `export_glm_tiny_qmoe.py`. Existing artifact
`/home/justinchu/glm-e2e-artifacts/glm-5.2-tiny-q4` contains 34 asymmetric
block-32 `MatMulNBits` nodes and passes:

```bash
GLM_TINY_Q4_E2E_DIR=/home/justinchu/glm-e2e-artifacts/glm-5.2-tiny-q4 \
cargo test -p onnx-genai-engine --test glm_tiny_quant_e2e \
  -- --ignored --nocapture
```

Result: `1 passed`, eight generated tokens. Real GLM-5.2 checkpoint packing
and practical sparse MoE emission remain separate work.

## Architecture gaps

| Model class | Quirk | CUDA/native status | Consequence |
|---|---|---|---|
| GLM-4 dense | partial interleaved RoPE (`0.5`), attention projection biases, fused gate/up, four RMS norms | **Supported.** Standard RoPE has `rotary_embedding_dim`; MatMul bias can remain `Add`; Split/Silu/Mul and RMS norms are covered. Mobius models these in `models/glm.py`. | Tractable now once a compatible quantized checkpoint is supplied. ORT GenAI's legacy ChatGLM builder may not match newer `Glm4ForCausalLM`; Mobius is the safer graph emitter. |
| DeepSeek-Coder 1.3B/6.7B dense | Llama-style MHA, RMSNorm, RoPE; no MLA or MoE | **Supported and proven** for 1.3B. | Best first production smoke target. |
| DeepSeek-V2/V3 | MLA: Q/K head 192, V head 128; partial RoPE only on 64 Q/K channels; low-rank Q/KV projections | **Functionally expressible.** Mobius decomposes MLA into standard ops plus opset-24 `Attention`; CUDA standard Attention explicitly supports a distinct `v_head_size`. | MLA itself is not the primary blocker. A real export and parity test are still required. |
| DeepSeek-V2/V3 | routed + shared MoE, group-limited/noaux routing | `QMoE` CUDA exists, but real Mobius checkpoint packing is not complete. Per-expert `MatMulNBits` unrolling is structurally possible but evaluates/masks every expert and is impractical. `BlockQuantizedMoE` is not registered in the CUDA EP. | Practical V2/V3 is blocked on a real fused `QMoE` emitter/packer (or CUDA `BlockQuantizedMoE`), not on int4 GEMV. |
| DeepSeek-V4 / GLM-5.2 | CSA/IndexShare selected-token attention | `CompressedSparseAttention`, `SparseKvGather`, and `IndexShare` are registered. Existing decision logs still classify the full real-model path as unverified; IndexShare capture and sparse-performance work remain. | Not a first smoke target. |
| DeepSeek-V4 / GLM-5.2 | MoE plus MTP/NextN state | Generic MTP hidden/state threading exists in engine code, but the model-specific indexer/cache state progression and real sidecar E2E remain pending (`mtp-state-threading` backlog). | Production speculative decoding remains blocked even after base weights export. |

## Ranked next steps

1. **Promote the generated DeepSeek-Coder-1.3B artifact as the first
   reproducible manual smoke target (done).**
   It is the smallest true DeepSeek checkpoint, uses only proven dense
   Llama-style ops, and already runs on native CUDA with zero fallbacks.
2. **Generate GLM-4-9B int4 through Mobius plus an external RTN/GPTQ/AWQ
   quantization step (2-4 days).** Graph support is tractable; the missing piece
   is a production packed checkpoint, because Mobius's CLI is an emitter/repacker,
   not a float-to-int4 quantizer.
3. **Run a real DeepSeek-V2-Lite/V3 MLA float or per-expert-int4 structural
   parity export (3-5 days).** This validates MLA shapes and partial RoPE while
   explicitly accepting that unrolled MoE is not performant.
4. **Finish real fused-QMoE packing in Mobius (1-2 weeks).** Required for a
   practical V2/V3 or GLM MoE package; validate expert-major packing against
   the CUDA `QMoE` ABI.
5. **Only then onboard GLM-5.2 / DeepSeek-V4 (multi-week).** Requires real CSA /
   IndexShare artifacts, sparse-path E2E, capture work, and model-specific MTP
   state/indexer-cache orchestration.

The truly blocked items are practical MoE packing and V4/GLM-5.2 sparse+MTP
orchestration. Dense DeepSeek-Coder and dense GLM-4 graph execution are
tractable now.
