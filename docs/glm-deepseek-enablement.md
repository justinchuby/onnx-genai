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
2. **Use DeepSeek-R1-Distill-Qwen-1.5B as the reasoning-model smoke
   target (done).** Its Qwen2 graph captures as one segment with zero fallbacks.
3. **Implement quantization-aware GLM-4 preprocessing in Mobius and emit the
   downloaded GPTQ checkpoint (done, pending Mobius PR #424).** The importer now
   remaps the legacy encoder hierarchy, preserves fused packed gate/up, and
   splits fused packed QKV into separate MatMulNBits projections.
4. **Run a real DeepSeek-V2-Lite/V3 MLA float or per-expert-int4 structural
   parity export (3-5 days).** This validates MLA shapes and partial RoPE while
   explicitly accepting that unrolled MoE is not performant.
5. **Finish real fused-QMoE packing in Mobius (1-2 weeks).** Required for a
   practical V2/V3 or GLM MoE package; validate expert-major packing against
   the CUDA `QMoE` ABI.
6. **Only then onboard GLM-5.2 / DeepSeek-V4 (multi-week).** Requires real CSA /
   IndexShare artifacts, sparse-path E2E, capture work, and model-specific MTP
   state/indexer-cache orchestration.

The truly blocked items are practical MoE packing and V4/GLM-5.2 sparse+MTP
orchestration. Dense DeepSeek-Coder, R1-Distill, and GLM-4-9B now run on the
native CUDA EP.

## Weight generation progress 2026-07-23

### DeepSeek-R1-Distill-Qwen-1.5B: runnable

ORT GenAI successfully exported the Qwen2-based reasoning distill:

```bash
python3 -m onnxruntime_genai.models.builder \
  -m deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  -o /home/justinchu/glm-e2e-artifacts/deepseek-r1-distill-qwen-1.5b-int4-cuda \
  -p int4 -e cuda \
  --extra_options int4_block_size=32 int4_is_symmetric=true \
                  int4_accuracy_level=0 hf_token=false
```

The 1.3 GB package contains 141 symmetric block-32 `MatMulNBits`, 28 GQA,
56 skip-RMSNorm, and no uncovered operator. Checksums:

```text
378380244262cc9c297f49161647a2e2e11c7be83b4657e82df1d2d13f602666  model.onnx
51064450ed6f55b5c2337bf25305f932d724d922c5eb1eb2a08b527764540485  model.onnx.data
b9ab37b59318bc067b9516dceccff1d9d15148ac9e4aa4e7ff4fa74990a42750  genai_config.json
```

Native CUDA on GPU 4, with graph capture enabled:

```bash
CUDA_VISIBLE_DEVICES=4 cargo run -q -p onnx-genai-bench \
  --features bench-native,cuda --bin profile_native -- \
  --model /home/justinchu/glm-e2e-artifacts/deepseek-r1-distill-qwen-1.5b-int4-cuda \
  --ep cuda --steady --warmups 1 --runs 1 --tokens 32 \
  --prompt 'Solve: what is 17*23? Think step by step.'
```

Steady decode was 549.88 tok/s (1.819 ms/token, 24 timed tokens after an
eight-token skip). A diagnostic non-steady run reported one captured segment,
zero eager seams, and `captures=1 replays=29 fallbacks=0` during measurement.
The greedy output was grammatical but degenerated into repeated
`"Show all your thinking"` text and did not answer the arithmetic problem.
Thus execution/capture is proven, but this plain-prompt greedy smoke is not a
quality validation.

### GLM-4-9B: exporter weight mapping resolved

The ORT GenAI builder does recognize `zai-org/GLM-4-9B` as `ChatGLMModel` and
selects GQA/int4 CUDA. It downloaded the ten-file checkpoint, then failed while
instantiating the current remote model code:

```text
AttributeError: 'ChatGLMConfig' object has no attribute 'max_length'.
Did you mean: 'seq_length'?
```

Therefore B1 is not blocked on architecture dispatch; it is blocked by
incompatibility between the builder's Transformers/config loading path and the
current GLM-4 remote implementation.

For B2, two public, ungated int4 checkpoints were downloaded:

| Checkpoint | Local source | Format / size |
|---|---|---|
| `jfiekdjdk/glm-4-9b-chat-awq` | `/home/justinchu/glm-e2e-artifacts/glm-4-9b-chat-awq-source` | AWQ asymmetric int4, group 128, 6.3 GB, PyTorch `.bin` shards |
| `ModelCloud/glm-4-9b-gptq-4bit` | `/home/justinchu/glm-e2e-artifacts/glm-4-9b-gptq-source` | GPTQ symmetric int4, group 128, 6.3 GB, one safetensors file |

The unmodified Mobius invocations fail on missing normalized metadata:
`num_hidden_layers must be positive, got 0` for the GPTQ config (which exposes
`num_layers=40`), and `hidden_act is None` for AWQ. After supplying only those
equivalent config fields (`num_hidden_layers=40`, `hidden_act=silu`) to isolate
the next failure:

- the AWQ local path is rejected because Mobius's local loader accepts only
  safetensors, while this checkpoint is stored as `.bin`;
- the GPTQ safetensors load succeeds, but package validation reports:

```text
ValueError: Component 'model' has 643 initializer(s) without weights:
'model.embed_tokens.weight', 'model.layers.0.input_layernorm.weight',
'model.layers.0.self_attn.q_proj.weight',
'model.layers.0.self_attn.q_proj.scales',
'model.layers.0.self_attn.k_proj.weight' (and 638 more).
Ensure all weights are loaded before saving.
Check if the preprocess_weights logic is correct.
```

The source checkpoint uses names such as
`transformer.encoder.layers.0.self_attention.query_key_value.qweight` and a
fused concatenated QKV projection. Mobius's previous ChatGLM preprocessing only
renames `self_attention` and MLP projection names; it does not remap the
`transformer.encoder` hierarchy and split/repack the fused quantized QKV into
the graph's `q_proj`/`k_proj`/`v_proj` initializers.

Mobius PR [onnxruntime/mobius#424](https://github.com/onnxruntime/mobius/pull/424)
implements that preprocessing. It also normalizes legacy ChatGLM configuration
fields (`num_layers`, `multi_query_group_num`, `kv_channels`, `seq_length`,
projection bias flags), selects the fused gate/up MLP, and preserves partial
interleaved RoPE (`0.5`).

The previously downloaded GPTQ safetensors checkpoint now emits successfully:

```bash
/home/justinchu/mobius/.venv/bin/mobius build \
  --model ModelCloud/glm-4-9b-gptq-4bit --trust-remote-code \
  --dtype f16 --ep cuda --runtime onnx-genai \
  --optimize=group_query_attention,skip_norm \
  /home/justinchu/glm-e2e-artifacts/glm-4-9b-int4-cuda
```

The 6.3 GB graph contains 240 symmetric block-128 `MatMulNBits`, 40 GQA,
80 skip-RMSNorm, and 40 activation `Split` nodes. Model checksums:

```text
8901dba568ae59b8caca0186f391350d4657af99c3c2656495fc83bc48d33033  model.onnx
8169705cdb528915bec69e82442123fbd40c4ea8eee9ef1558264bfb736d394b  model.onnx.data
54323f454c7eddebf2e46a5296d9df2fef6fde609d5e8185ef1d3aa418e8272a  inference_metadata.yaml
```

Native CUDA on GPU 4 generated a coherent response:

```text
" I'm a bot. I'm trying to help you with your question.
, but I'm not sure what you're asking. Can you please clarify your question"
```

Steady decode was 91.51 tok/s (10.928 ms/token). Capture diagnostics reported
41 captured segments and 40 eager seams, one per fused-MLP activation `Split`.
The measured run reported `captures=1 replays=29 fallbacks=0`, so GLM-4 dense
execution is unblocked. The seams are a performance issue, not an execution
fallback.

One packaging caveat remains: the original GPTQ repository exposes only a
custom slow tiktoken tokenizer, so Mobius cannot derive `tokenizer.json`
automatically. The smoke package uses the compatible fast tokenizer from
`THUDM/glm-4-9b-chat-hf`. This does not block graph execution, but tokenizer
fallback/source selection should be automated before distributing the package.

### GLM-4 capture fragmentation

The 40 seams are one fused-MLP activation split per decoder layer; they are not
QKV or partial-RoPE splits. For example, layer 0 contains:

```text
MatMulNBits gate_up_proj: [batch, sequence_len, 27392]
  -> Split(axis=-1, num_outputs=2)
     -> gate: [batch, sequence_len, 13696] -> Sigmoid -> Mul
     -> up:   [batch, sequence_len, 13696] ------------> Mul
  -> MatMulNBits down_proj
```

The node is `model/layers.0/mlp/Split_node_24`; layers 1 through 39 repeat the
same pattern, ending at `model/layers.39/mlp/Split_node_726`. Each Split has one
data input, no runtime split-size tensor, `axis=-1`, and `num_outputs=2`.
Mobius emits it in `FusedGateUpMLP.forward`
(`src/mobius/components/_mlp.py:138-144`).

The shapes are already resolved. Capture diagnostics classify every Split as
`KernelCaptureUnsupported`, not `UnresolvedInputShape`:

```text
Split reads runtime split sizes on the host and performs a trailing stream synchronization
```

The CUDA Split kernel launches one copy kernel per output and then calls
`runtime.synchronize()` (`movement.rs:923-1018`). Its `capture_support()`
therefore rejects capture unconditionally. The executor merely partitions
around that kernel (`executor.rs:2785-2907`); the earlier Phi shape seeding is
not applicable. Phi had unresolved shapes downstream of a control-flow `If`;
GLM's Split input and both output shapes are concrete before capture planning.

Measured on GPU 4:

- captured, 41 segments: 10.928 ms/token, 91.51 tok/s;
- graph disabled: 22.272 ms/token, 44.90 tok/s;
- an instrumented captured decode charged the 40 eager Splits 3.037 ms total
  (about 76 microseconds each), before accounting for the extra graph replay
  calls. The profiler perturbs absolute timing, so this is directional.

Two implementation avenues:

1. **Mobius-side packed gate/up split (recommended first, 0.5-1 day).**
   Split `gate_up_proj.{weight,scales,bias}` along the already-repacked output
   dimension during ChatGLM checkpoint preprocessing, emit separate
   `gate_proj` and `up_proj` MatMulNBits nodes, and use the standard MLP graph.
   This removes all 40 runtime Splits and should produce one captured segment.
   The transformation is layer-count-independent and analogous to the proven
   packed-QKV output split. It should be bit-exact because output columns are
   independent. Cost: 40 additional captured MatMulNBits launches. The existing
   paired gate/up SwiGLU CUDA fusion only accepts block-32, so it will not fuse
   this checkpoint's block-128 projections.

   Replacing Split with Slice is not useful today: CUDA Slice also reads bounds
   on the host, allocates metadata, synchronizes, and rejects capture
   (`movement.rs:588-713`).

2. **CUDA Split static capture path (1-2 days, broader but riskier).**
   This is a kernel change, not executor shape seeding. Teach `SplitFactory` to
   recognize the one-input static/even-split form, precompute the axis sizes,
   omit the trailing synchronization, and return capture support only for that
   validated form. Keep two-input/data-dependent Split unsupported. This also
   collapses GLM to one segment while retaining the single fused gate/up GEMV,
   and benefits other static-Split graphs. It needs capture/eager parity,
   prefill/decode shape, dynamic-split rejection, and token-exact tests because
   it changes stream-lifetime behavior in the CUDA movement kernel.

The Mobius change gives the same expected 41-to-1 segment collapse with a
smaller correctness blast radius, so it should be prototyped first on the
existing GLM import PR. The prior Phi 35-to-3 collapse measured only about 2.3%
from removing downstream graph replay calls; GLM additionally pays 40 explicit
stream synchronizations. Conservatively, removing the seams should move
91.5 tok/s into roughly the 110-125 tok/s range (20-37%), with the static CUDA
Split path likely owning the higher end because it preserves one gate/up GEMV.
