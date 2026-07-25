# ONNX Model Metadata Convention

Our runtime reads inline metadata from ONNX models using the `onnx_runtime.` namespace prefix.
This provides a fourth (lowest-priority) source of execution hints — embedded directly in the
model graph by the model author or export tool (e.g. Mobius).

## Namespace

All metadata keys use the prefix: **`onnx_runtime.`**

This avoids collision with:
- `onnx.` (reserved by ONNX spec)
- `com.microsoft.` (ORT internal)
- Other runtime-specific namespaces

## Where Metadata Lives in ONNX

ONNX protobuf has `metadata_props` at multiple levels:

```protobuf
// ModelProto.metadata_props — model-level
message ModelProto {
  repeated StringStringEntryProto metadata_props = 14;
}

// GraphProto — graph-level (rarely used)
// NodeProto.metadata_props — per-node (added in ONNX IR 10 / opset 21+)
message NodeProto {
  repeated StringStringEntryProto metadata_props = 16;  // IR version 10+
}
```

For models using older IR versions (< 10), node-level metadata is unavailable.
Use `doc_string` as fallback or external `execution_hints.json`.

## Metadata Keys

### Node-Level (`NodeProto.metadata_props`)

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `onnx_runtime.device` | string | Preferred device for this node | `"gpu"`, `"gpu:0"`, `"cpu"`, `"npu"` |
| `onnx_runtime.device.strength` | string | Hint strength | `"prefer"` (default), `"force"` |
| `onnx_runtime.memory.pin` | bool | Pin output tensors of this node | `"true"` |
| `onnx_runtime.memory.priority` | string | Eviction priority | `"high"` (pin), `"low"` (evict first), `"normal"` |
| `onnx_runtime.scheduling.cuda_graph` | bool | Include in CUDA graph capture region | `"true"`, `"false"` |
| `onnx_runtime.scheduling.overlap` | bool | Allow overlap with adjacent ops | `"true"` |
| `onnx_runtime.group` | string | Colocation group name — nodes with same group stay on same device | `"attention_block_0"` |
| `onnx_runtime.layer` | int | Logical layer index (for layer-range hints) | `"0"`, `"31"` |
| `onnx_runtime.offloadable` | bool | This node can be offloaded to CPU when GPU is full | `"true"` |
| `onnx_runtime.kernel` | string | Preferred kernel implementation | `"flash_attention"`, `"cutlass"` |

### Graph-Level (`GraphProto.metadata_props` or `ModelProto.metadata_props`)

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `onnx_runtime.model.num_layers` | int | Total transformer layers (enables layer-range hints) | `"32"` |
| `onnx_runtime.model.layer_pattern` | string | Naming pattern for layer nodes | `"layers.{}.attention"`, `"model.layers.{}"` |
| `onnx_runtime.model.architecture` | string | Model architecture hint | `"llama"`, `"phi"`, `"gemma"` |
| `onnx_runtime.memory.arena_gpu_mb` | int | Suggested GPU arena size (MB) | `"4096"` |
| `onnx_runtime.memory.arena_cpu_mb` | int | Suggested CPU arena size (MB) | `"8192"` |
| `onnx_runtime.memory.prefetch` | string | Comma-separated tensor names to prefetch | `"embed_tokens.weight,lm_head.weight"` |
| `onnx_runtime.version` | string | Metadata schema version | `"1"` |

### Layer-Based Hints (using `onnx_runtime.layer`)

When nodes are annotated with `onnx_runtime.layer`, the runtime can apply layer-range
placement from `execution_hints.json` without pattern matching on node names:

```json
{
  "placement": [
    {
      "selector": { "layer_range": { "start": 0, "end": 7 } },
      "device": { "type": "gpu", "index": 0 },
      "strength": "force"
    },
    {
      "selector": { "layer_range": { "start": 8, "end": 15 } },
      "device": { "type": "gpu", "index": 1 },
      "strength": "force"
    },
    {
      "selector": { "layer_range": { "start": 16, "end": 31 } },
      "device": { "type": "cpu" },
      "strength": "prefer",
      "reason": "Offload last 16 layers to CPU for 16GB GPU"
    }
  ]
}
```

The runtime resolves `layer_range` by reading `onnx_runtime.layer` from each node.
If nodes don't have this annotation, it falls back to `onnx_runtime.layer_pattern`
to infer layer index from node names.

## Example: Mobius-Generated Model

A Llama-3 model exported by Mobius might have:

```
ModelProto.metadata_props:
  onnx_runtime.version = "1"
  onnx_runtime.model.num_layers = "32"
  onnx_runtime.model.layer_pattern = "model.layers.{}"

Node "model.layers.0.self_attn.q_proj" metadata_props:
  onnx_runtime.layer = "0"
  onnx_runtime.group = "attn_0"

Node "model.layers.0.self_attn.GroupQueryAttention" metadata_props:
  onnx_runtime.layer = "0"
  onnx_runtime.group = "attn_0"
  onnx_runtime.device = "gpu"
  onnx_runtime.device.strength = "force"
  onnx_runtime.kernel = "flash_attention"
  onnx_runtime.scheduling.cuda_graph = "true"

Node "model.layers.0.mlp.gate_proj" metadata_props:
  onnx_runtime.layer = "0"
  onnx_runtime.offloadable = "true"

Node "model.embed_tokens.Gather" metadata_props:
  onnx_runtime.device = "cpu"
  onnx_runtime.device.strength = "prefer"
  onnx_runtime.memory.priority = "low"
```

## Priority Resolution

When the same node gets hints from multiple sources:

```
Priority (highest → lowest):
1. Programmatic builder API (.placement_hint(...))
2. execution_hints.json (user file)
3. inference_metadata.yaml → execution_hints section
4. ONNX model metadata_props (onnx_runtime.* keys)  ← lowest
```

For conflicting strengths:
- `force` from any source = always respected (error if contradicting forces)
- `prefer` from higher-priority source overrides lower-priority `prefer`

## Attention key-sequence lengths

The canonical `com.microsoft::GroupQueryAttention` key-sequence-length metadata
is a contiguous `int32 [batch_size]` tensor. Model packages whose graph
structurally produces a rank-0 value for unit-batch decode may explicitly
declare:

```yaml
model:
  attention:
    type: group_query_attention
    key_sequence_lengths:
      scalar_broadcast: unit_batch
```

This permits only a contiguous rank-0, one-element `int32` tensor when the
executing attention batch is exactly one. It does not authorize broadcasting a
single value across a multi-row or ragged batch; batch sizes greater than one
still require the canonical `[batch_size]` representation. Omitting the field
keeps strict canonical validation.

## Native decoder and proposer execution contracts

`model.io` (target decoder) and `speculative.io` (standalone proposer) use the
same explicit, model-agnostic contract:

- `sequence_source`: `token_ids` or `inputs_embeds`.
- `token_input` / `inputs_embeds_input`: exact graph port selected by
  `sequence_source`.
- `kv_ownership`: `owned` when the graph carries positional
  `kv_inputs`/`kv_outputs`, or `shared` when it references target-owned cache.
- `logits_output`: output carrying token scores.
- `hidden_output`: output carrying a hidden/projected recurrent state.

Absent `sequence_source` and `kv_ownership` preserve the historical target and
ordinary draft-model defaults: `token_ids` plus `owned` KV. A shared-KV
proposer should declare all axes explicitly:

```yaml
model:
  io:
    sequence_source: token_ids
    kv_ownership: owned
    token_input: input_ids
    attention_mask_input: attention_mask
    position_ids_input: position_ids
    logits_output: logits
    hidden_output: target_hidden
    kv_inputs: [past.0.key, past.0.value]
    kv_outputs: [present.0.key, present.0.value]

speculative:
  proposal_type: shared_kv
  model: assistant/model.onnx
  io:
    sequence_source: inputs_embeds
    kv_ownership: shared
    inputs_embeds_input: proposer_embeddings
    attention_mask_input: proposer_mask
    position_ids_input: proposer_positions
    logits_output: draft_logits
    hidden_output: projected_state
  shared_kv:
    - name: attention_group
      key_input: assistant_cache.key
      value_input: assistant_cache.value
      target_key_input: past.0.key
      target_value_input: past.0.value
```

Shared-KV port names are data. The runtime does not derive them from a model
family, hidden size, or tensor-name convention. Native CPU execution supplies
the target decoder's carried cache tensors to the proposer step; native CUDA
device-buffer aliasing is a separate capability and fails with an actionable
error rather than falling back to ORT.

## Colocation Groups

Nodes with the same `onnx_runtime.group` value are treated as a colocation set.
The ILP solver adds constraints that all nodes in a group must map to the same device.

Typical use: attention Q/K/V projections + attention kernel + output projection
all need to be on the same device (to avoid cross-device data movement for KV cache).

```
# All nodes with group="attn_0" → same device
onnx_runtime.group = "attn_0"
```

This is equivalent to a `ColocateHint` in `execution_hints.json` but embedded in the model.

## Validation

On model load, the runtime:
1. Scans all `onnx_runtime.*` keys
2. Warns on unrecognized keys (typo detection)
3. Validates value types (e.g. `onnx_runtime.layer` must be parseable as int)
4. Reports conflicting `force` hints as hard errors

```rust
pub enum MetadataWarning {
    UnknownKey { node: String, key: String },
    InvalidValue { node: String, key: String, value: String, expected: &'static str },
    ConflictingForce { node: String, source_a: HintSource, source_b: HintSource },
}
```

## Multimodal input: the placeholder contract

A vision-language package declares two things, and the runtime derives everything
else from them. Neither is inferred from a model or vendor name.

**1. How pixels become a tensor** — `preprocessing.image`, a transform program
(decode → resize → rescale → normalize → tile/patchify) whose `outputs` bind
produced tensors to exact `component.input` endpoints:

```yaml
preprocessing:
  image:
    transforms:
      - {op: decode, outputs: [decoded]}
      - {op: convert_rgb, inputs: [decoded], outputs: [rgb]}
      - {op: resize, inputs: [rgb], outputs: [resized], size: 336, mode: fixed}
      - {op: rescale, inputs: [resized], outputs: [rescaled], scale: 0.00392156862745098}
      - {op: normalize, inputs: [rescaled], outputs: [normalized], mean: [0.5, 0.5, 0.5], std: [0.5, 0.5, 0.5]}
    outputs:
      - {source: normalized, name: vision_encoder.pixel_values, content: pixels, dtype: float32}
```

**2. Where the image sits in the text** — `pipeline.vision`, the placeholder
expansion contract:

```yaml
pipeline:
  vision:
    image_placeholder_token_id: 262144   # marks WHERE to expand
    image_token_id: 262145               # what is written into the expansion
    token_count_source: per_tile         # per_tile | per_patch | from_grid
    tokens_per_tile: 256
```

The prompt must carry exactly one `image_placeholder_token_id` per image, in
prompt order. Before KV sizing, each placeholder is replaced by the declared run
of `image_token_id`, whose length comes from `token_count_source`.

### Callers do not write placeholders by hand

Requiring a caller to type a model's private placeholder spelling would mean
reading this file to write a prompt. Both front ends therefore insert them:

- a prompt that already positions placeholders is honored verbatim;
- a prompt that positions none gets one per image **prepended**, in the
  conventional "images, then the question about them" order;
- a *partial* set is rejected. The caller clearly meant to position them, and
  guessing where the rest belong would silently change which image a sentence
  refers to.

The rule lives in `onnx_genai_server::multimodal`, shared by the CLI and
`/v1/chat/completions`.

### Audio input: two different shapes

Audio arrives in one of two shapes, and only the first is implemented today.

**Encoder-decoder speech recognition** (Whisper-style) — supported. A component
declares an `input_features` input; the runtime extracts log-mel features into
it and seeds the decoder with the model's own transcription prompt. The spoken
audio *is* the content, so it replaces the text prompt rather than sitting
inside it. This is what `onnx-genai transcribe`, `generate --audio`, and
`/v1/audio/transcriptions` drive.

**Audio as an embedded modality in an omni LLM** (Gemma-3n/4-style, where audio
tokens are interleaved with text like image tokens) — **not yet implemented**.
Structurally it is the vision contract with a different encoder: it needs an
`audio_placeholder_token_id` / `audio_token_id` expansion contract alongside
`pipeline.vision`, plus a `preprocessing.audio` program binding features to the
encoder endpoint. The engine's dataflow already expresses the graph; what is
missing is the declared contract and its expansion, not the execution. Adding it
should generalize the existing expansion rather than duplicate it, so a package
declares *modality → placeholder → token count* once per modality.

### Prefix caching and multimodal prompts

The prefix cache is a **token-id trie** keyed on the prompt's tokens, and it is
attached to the single-model `Engine`. Multi-component pipelines — every
multimodal path — run on `PipelineEngine`, which does not consult it, so a
multimodal turn always reports `prefix_cache_hit_len == 0`.

That is the safe default, and enabling it naively would be a **correctness bug**:
two different images expand to the *same* run of `image_token_id`, so a
token-only key would happily reuse KV pages computed from a different image's
embeddings — the model would attend to the wrong picture. Extending prefix
caching to multimodal prompts requires folding a digest of the injected
multimodal tensors into the cache key, so that identical text with a different
image misses.

Text-only prompts do benefit today, including across CLI REPL turns: a second
turn reuses the first turn's prompt *and* its generated tokens.
