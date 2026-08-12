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

For a bare decoder-only document, `model.io` is the target decoder contract.
For a composite document, each graph declares its contract only at
`pipeline.models.<component>.io`; top-level `model.io` is invalid when
`pipeline` is present. `speculative.io` remains the standalone proposer
contract. All three locations use the same explicit, model-agnostic fields:

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

I/O role resolution follows one reusable priority order: an exact `model.io` or
`speculative.io` declaration, then a unique tensor-shape signal. Graph port
names are never interpreted. When shapes are ambiguous—for example, token IDs,
attention masks, and position IDs all having rank two, or multiple same-shaped
KV tensors—the corresponding metadata fields are required and loading fails
with an error naming the missing field.

### Graph-derived `model.io` fallback for hybrid decoders (#384)

When a package declares **no** `model.io` block at all, the native loader
attempts one additive fallback before shape inference: it derives the decoder
I/O topology from the graph's port inventory using the conventional
onnxruntime-genai key/value names (`past_key_values.%d.key`/`.value` →
`present.%d.key`/`.value`). This engages **only** when the derivation finds at
least one recurrent state pair — i.e. a hybrid linear-attention decoder that
also exposes non-KV `conv_state`/`recurrent_state` ports (qwen3.x, incl. the
27B). Those ports carry fixed loop-carried state that shape inference cannot
classify, so without this fallback such stock exports fail to load natively.

A declared `model.io` always wins; the fallback never overrides one. Pure-dense
decoders derive zero state pairs, so the fallback declines and they keep the
existing shape-inference path unchanged — no model that loads today changes its
behavior. The derivation reuses the same guarded logic as the
`genai_config.json` compatibility path (recurrent ports are matched by suffix,
never misclassified as KV), with one relaxation: a `present.*` state port whose
exported shape is fully symbolic is accepted against its concrete `past_*` input
(a symbolic axis is unknown, not proof of a different shape).

These hybrids run **eager**: their `Scan` control-flow body declines CUDA-graph
capture, so this fallback unblocks correctness (native decode of the model
class) independent of the capture perf lane. Correctness is validated
byte-for-byte against the native CPU fp32 oracle (ORT-CUDA crashes on this model
class, so there is no ORT reference).

Attention representation metadata is interpreted independently of
`model.attention.type`. In particular, `key_sequence_lengths.scalar_broadcast`
controls the representation contract rather than selecting an implementation,
and `model.io.kv_update: shared_buffer` declares fixed-capacity KV behavior
without requiring a particular attention operator name. Omitting the scalar
permission keeps the canonical vector requirement.

### Generating `inference_metadata.yaml` from a `genai_config.json` (#384)

A stock onnxruntime-genai export ships `genai_config.json` but **not**
`inference_metadata.yaml`, so it is not native-loadable until the metadata file
exists — the #384 gap. When the export is a plain single-decoder LLM (the
`model.io` block is fully derivable from the genai config's declared ports),
`scripts/gen_inference_metadata.py` emits the file deterministically:

```bash
python scripts/gen_inference_metadata.py MODEL_DIR            # writes MODEL_DIR/inference_metadata.yaml
python scripts/gen_inference_metadata.py MODEL_DIR --stdout   # print without writing
python scripts/gen_inference_metadata.py MODEL_DIR --force    # overwrite an existing file
```

It reads only `genai_config.json` and maps `model.context_length →
model.max_sequence_length`; `model.decoder.inputs.input_ids / attention_mask →
io.token_input / io.attention_mask_input`; `model.decoder.outputs.logits →
io.logits_output`; and expands the `past_key_names` / `past_value_names` /
`present_key_names` / `present_value_names` `%d` templates over
`num_hidden_layers` into positionally paired `io.kv_inputs` / `io.kv_outputs`
(key-then-value per layer). This turns a one-off hand-written fix into a
repeatable step; it was used to make the qwen14b export native-loadable for the
KV-floor measurement
(`crates/onnx-genai-engine/tests/qwen14b_kv_floor_sweep_native_cuda.rs`), and it
reproduces that hand-written file exactly apart from a two-line provenance
comment header that the generator prepends (verified: the emitted YAML is
identical to the hand-written qwen14b-zp `inference_metadata.yaml` once the
header is dropped and line endings are normalized). It intentionally covers
**only**
the derivable single-decoder case: hybrid linear-attention decoders with
recurrent state ports still rely on the graph-derived `model.io` fallback above,
and any package that already declares `model.io` should keep its authoritative
declaration.

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

This scanning, type validation, and priority resolution is implemented in
`onnx_std::metadata_hints`. `MetadataHints::from_model` reads the embedded
`onnx_runtime.*` metadata off a loaded model; `MetadataHints::scan` is the
source-agnostic entry point that merges hints from any mix of [`HintSource`]s
through the same validation and priority logic.

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

An image costs a turn twice: the encoder forward pass, and a prompt in which
that one image has expanded into hundreds or thousands of placeholder tokens.
A conversation that keeps referring to the same picture would pay both on every
turn. `PipelineEngine` removes both repeats.

**Why the single-model prefix cache cannot simply be reused.** That cache is a
**token-id trie**, and a token-id key is unsound the moment embeddings enter the
prompt from anywhere but the token embedding table. Placeholder expansion
replaces an image with a run of one repeated `image_token_id`, so two entirely
different photographs produce *byte-identical* token sequences. Keyed on tokens
alone, a cache would serve the first photograph's KV for the second, and the
model would answer fluently about a picture it was never shown.

**The key is therefore tokens plus a digest of every bound input tensor.** The
digest is a 128-bit content hash over each `component.input` endpoint's dtype,
shape, and element bytes. Change the picture and the digest changes, the prefix
stops matching, and the turn is recomputed. 128 bits rather than 64 because
nothing verifies a hit.

Two reuses follow from that key:

* **Encoder memoization.** A prompt-phase component is a pure function of its
  inputs — that is what distinguishes it from an `every_step` component — so its
  outputs are memoized under the digest of those inputs. Re-asking about the
  same attachment costs a hash instead of a vision or audio encoder pass. The
  budget is `EngineConfig::pipeline_cache_bytes` (default 512 MiB, `0` disables);
  entries are evicted least-recently-used.
* **Decoder KV prefix reuse.** The decoder keeps the KV from the previous
  generation and prefills only the tokens the new prompt added.

Reuse covers the **common prefix**, so a prompt that diverges part-way still
keeps the head it shares — which is what makes forking a conversation, editing
an earlier turn, or replaying a reasoning model's history (with the thinking the
KV still holds stripped out) reuse anything at all. Divergence requires
truncating the retained KV, and a pipeline component's past is an opaque
per-graph tensor with no declared sequence axis, so the axis is identified by
extent: the one dimension equal to the current KV length. Truncation declines,
and the turn recomputes, when

* more than one axis matches, since choosing between them is a guess and
  guessing wrong corrupts attention silently;
* the decoder carries fixed loop-carried state, which advances with the sequence
  but exposes no position to rewind to; or
* position ids are not a plain `linear_increment` of the absolute past length,
  because a carried or externally supplied coordinate would resume from
  positions describing tokens that no longer exist.

Reuse is also skipped entirely when the decoder's position ids arrive over a
`dataflow` edge, since such a tensor covers the whole prompt and prefilling only
a suffix would hand the decoder positions for tokens it is not being given.

When the decoder's `present.*` outputs describe a layout the page table can
address, the KV is **paged** — the same reference-counted page table, radix
prefix trie, and copy-on-write sharing the single-model engine uses. Many
prefixes are then held at once, so interleaved conversations do not evict each
other: several agents running under one long system prompt hold *one* copy of
its KV between them, and a conversation resumed after others have run still
finds its own tail.

Sharing is **page-granular**. The trie only reports a match where something was
published, so each finished generation is published at every page boundary as
well as at its full length; a prompt that diverges then matches up to the last
page they had in common. A prefix shorter than one page cannot be shared, which
in practice only affects prompts shorter than `page_size`.

A decoder whose KV cannot be paged falls back to retaining a single context,
truncated to the common prefix. That serves a shared head across conversations
but not each conversation's own tail, since there is only one slot.

Memoization additionally requires the component's graph to contain only
deterministic operators. A declared phase says *when* a component runs, never
that it is pure, so purity is read off the graph rather than assumed: memoizing
a graph containing `RandomNormal` would freeze its first draw and return it
forever. Everything the runtime will execute is checked — subgraphs of `Loop`
and `If`, and model-local functions, whose calling node's `op_type` reveals
nothing about the body that gets inlined.

The check covers the standard ONNX operator set, where which operators are
random is fixed by the specification. A custom-domain operator is taken at face
value; a package whose encoder contains a non-deterministic custom operator
should set `pipeline_cache_bytes` to `0`.

Two further consequences worth knowing:

* **Reuse stops one token short of the previous turn's output.** The last
  sampled token is appended to the context but never fed back to the decoder, so
  no KV exists for it and the next turn prefills it.
* **Reasoning models reuse only up to their divergence.** Earlier turns'
  thinking is stripped before the conversation is replayed (see the CLI's
  multi-turn behavior), so the next prompt diverges from the retained context
  where the thinking began. Everything before that — including the whole
  expanded image — is still reused.

`--profile` reports both as `encoder cache` (hits / runs) and
`multimodal prefix reuse` (tokens carried over).

Text-only prompts continue to use the single-model token-id prefix cache,
including across CLI REPL turns: a second turn reuses the first turn's prompt
*and* its generated tokens.
