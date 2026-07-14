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
| `onnx_runtime.num_layers` | int | Total transformer layers (enables layer-range hints) | `"32"` |
| `onnx_runtime.layer_pattern` | string | Naming pattern for layer nodes | `"layers.{}.attention"`, `"model.layers.{}"` |
| `onnx_runtime.memory.arena_gpu_mb` | int | Suggested GPU arena size (MB) | `"4096"` |
| `onnx_runtime.memory.arena_cpu_mb` | int | Suggested CPU arena size (MB) | `"8192"` |
| `onnx_runtime.prefetch` | string | Comma-separated tensor names to prefetch | `"embed_tokens.weight,lm_head.weight"` |
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
  onnx_runtime.num_layers = "32"
  onnx_runtime.layer_pattern = "model.layers.{}"

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
