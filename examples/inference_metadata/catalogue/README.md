# Inference-metadata model catalogue

These are **C/design-level, config-only examples** of the canonical serialized ABI:
`pipeline.workflow`. They parse and pass `onnx_genai_metadata::validate_metadata`, but
their referenced `*.onnx.textproto` graphs are illustrative and are not checked in.
Accordingly, the catalogue is not numerical, performance, or end-to-end proof.

The schema has no generic model-name or model-source field. Identity and intended
provenance are therefore stated here and in YAML comments rather than invented as
non-schema keys.

| # | Example | Distinguishing contract |
|---|---|---|
| 1 | [Gemma 4 text decoder](01-gemma4-text-decoder.yaml) | Separate full- and sliding-attention KV groups |
| 2 | [Cosmos3 Edge rollout](02-cosmos3-edge-rollout.yaml) | Observation/action step with session-persistent world state |
| 3 | [Qwen3.5 VLM](03-qwen3_5-vlm.yaml) | Vision encoder → projector/embedding → decoder |
| 4 | [Whisper](04-whisper-encoder-decoder.yaml) | Audio encoder → cross-attending decoder |
| 5 | [Wav2Vec2 CTC](05-wav2vec2-ctc.yaml) | Encoder-only logits plus CTC decoding profile |
| 6 | [PersonaPlex](06-personaplex-full-duplex.yaml) | Full-duplex step with independent dialogue/codec state |
| 7 | [Stable Diffusion](07-stable-diffusion-text-to-image.yaml) | Text-conditioned denoising loop → image decoder |
| 8 | [Qwen Image Edit](08-qwen-image-edit.yaml) | Image-conditioned denoising loop |
| 9 | [CogVideoX](09-cogvideox-text-to-video.yaml) | Temporal latent loop → video output |
| 10 | [LoRA selection](10-lora-adapter-selection.yaml) | Request-aligned adapter segments/counts/scales |
| 11 | [Speculative decoding](11-speculative-proposer-verifier.yaml) | Explicit proposer → verifier dataflow |
| 12 | [ESM-2](12-esm2-protein-encoder.yaml) | Protein encoder, no persistent state |
| 13 | [ProtBert](13-protbert-protein-encoder.yaml) | Protein encoder, no persistent state |
| 14 | [WeatherNext rollout](14-weathernext-rollout.yaml) | Session-persistent atmospheric state |
| 15 | [Windowed attention](15-windowed-attention.yaml) | Stateless local attention vs stateful streaming window |
| 16 | [Linear attention](16-linear-attention-recurrent.yaml) | `recurrent` + `replace`, no `sequence_axis` |
| 17 | [Causal convolution](17-causal-convolution-recurrent.yaml) | Separate accumulator/history recurrent groups |
| 18 | [Static cache](18-static-cache-indexed-scatter.yaml) | Fixed capacity, logical lengths, indexed scatter |
| 19 | [Operator ABI comparison](19-operator-abi-comparison.yaml) | Graph-visible operator/port distinctions |
| 20 | [Qwen3.5 hybrid speculative decode](20-qwen3_5-hybrid-speculative-decoding.yaml) | Full-attention KV plus linear/conv replacement state with atomic rollback |
| 21 | [Shared-prefix pixel flow](21-shared-prefix-pixel-flow.yaml) | Alternating CFG branches read frozen prefix state across a flow-matching loop |

## Shared-prefix alternating branches

The shared-prefix example is architecture-neutral. An understanding component
advances conditional and unconditional state groups once. Generation components
then bind those groups with `access: read_only`, so alternating CFG branches can
reuse the frozen prefixes without pretending that discarded graph outputs are
state transitions. The same pattern applies to unified any-to-any architectures
whose later stages consume, but must not advance, state produced by an earlier
stage.

## Qwen3.5 hybrid state and speculative decoding

Qwen3.5-style hybrid decoders do not have one homogeneous cache. The example
declares three target-owned groups:

- attention KV is sequence-growing `full_attention` state with `update: append`;
- the linear-attention accumulator is fixed-size `recurrent` state with
  `update: replace` and no `sequence_axis`;
- causal-convolution history is another fixed-size `recurrent` replacement
  group because it has different shape and graph ports.

For a proposal of at most four positions, `speculative.rollback_state` names
every affected target state cell. Each group declares `rollback_positions: 4`,
and `cascade` makes the three groups one atomic rollback unit. Rejecting a suffix
therefore truncates the attention KV to the accepted length, but it **cannot**
slice either replacement tensor: the runtime must restore a per-prefix snapshot
or restore the pre-proposal snapshot and replay the accepted prefix. The
metadata declares that both strategies are legal through the rollback and
snapshot capabilities; it does not choose one.

The example uses an independent proposer whose state is recomputed from
committed tokens. A persistent draft model must declare its own KV, linear, and
convolution groups and add those cells to `rollback_state` as well. Mutable
target recurrent state must not be listed in `shared_state` merely to save
memory: it is shareable only when proposer and target genuinely use the same
graph-visible state ABI and the whole shared group obeys the same rollback
bound.

## Attention/operator ABI matrix

| Case | Graph-visible metadata fact | Runtime-private choice |
|---|---|---|
| ONNX `TensorScatter` / indexed scatter | State group declares `update.kind: indexed_scatter`, destination port, capacity, and logical length when the graph exposes them | Buffer allocator, storage placement, and slot policy |
| `com.microsoft::GroupQueryAttention` | Component artifact/opset and graph ports; state groups still describe the graph-visible cache ABI | Which equivalent kernel implements the node |
| ONNX `Attention` | Standard opset and graph ports/state bindings | Flash, memory-efficient, or unfused kernel dispatch |
| Explicit paged attention | A versioned component contract is appropriate only when graph inputs expose block tables/slot mappings; the example binds both | Page allocator and physical page placement |
| Runtime-private paged substitution | No paging metadata: the graph remains ordinary Attention with its ordinary state ports | Runtime may substitute paged storage/kernels while preserving that ABI |

Operator spelling alone does not create allocator or storage policy. Conversely, a
graph-visible block table, slot mapping, write index, or logical length cannot be
hidden as a kernel choice because it is part of the component's typed ABI.

## Port and heterogeneous KV semantics

ONNX component artifacts are authoritative for port dtype, rank, shape, and
opset imports, so an ONNX-backed component may omit duplicated
`ports.inputs`/`ports.outputs` contracts. It must still declare semantic
`ports.roles` for values such as token ids and logits that ONNX cannot
identify. State ports are classified separately by each state group's aliases:
`input`, `output`, `role: key|value|combined`, and numeric `layer`.

`layer` orders and pairs buffers; it does not impose geometry. The Gemma 4
example deliberately uses independent `full_key_heads`, `full_value_heads`,
`sliding_key_heads`, and `sliding_value_heads` dimensions. A runtime reads each
actual graph port independently, so layers may have different KV head counts
and a layer may even expose different K and V head counts. Split groups only
when update discipline, layout, dtype, sequence axis, lifetime, or rollback
semantics differ—not merely because two aliases have different shapes.

## Existing evidence

The separate [Qwen Image Edit evidence record](../evidence/qwen-image-edit-2509/README.md)
and [`qwen_image_edit_workflow_e2e.rs`](../../../crates/onnx-genai-engine/tests/qwen_image_edit_workflow_e2e.rs)
are real repository evidence; they do not turn this illustrative catalogue YAML into
an artifact-backed package. ComfyUI import behavior is documented and tested through
[`COMFYUI_IMPORT.md`](../../../docs/genai/COMFYUI_IMPORT.md) and the
`onnx-genai-comfyui-config` crate. No outputs or metrics are claimed here.
