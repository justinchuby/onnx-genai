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
| 22 | [Qwen3 chained speculative decode](22-qwen3-chained-speculative-decoding.yaml) | Token-embedding chain with typed hidden/KV recurrence and mapped vocabulary |
| 23 | [Gemma 4 E2B decoder](23-gemma4-e2b-decoder.yaml) | Dense hybrid full/sliding KV with heterogeneous global/local head widths and shared owners |
| 24 | [Gemma 4 E2B assistant](24-gemma4-e2b-assistant-speculative.yaml) | Cacheless read-only merged shared-KV drafter with a folded chained carry and graph-internal centroid pruning |
| 25 | [Gemma 4 26B-A4B MoE decoder](25-gemma4-26b-a4b-moe-decoder.yaml) | Sparse MoE (128 routed + 1 shared) with hybrid attention and heterogeneous global/local KV head count and width |

## Shared-prefix alternating branches

The shared-prefix example is architecture-neutral. An understanding component
advances conditional and unconditional state groups once. Generation components
then bind those groups with `access: read_only`, so alternating CFG branches can
reuse the frozen prefixes without pretending that discarded graph outputs are
state transitions. The same pattern applies to unified any-to-any architectures
whose later stages consume, but must not advance, state produced by an earlier
stage. Conditional and unconditional prefixes may have different sequence
lengths. The workflow derives resolution-aware initial noise from semantic
seed/width/height inputs, while retaining an optional caller-supplied latent
for controlled parity runs, and clamps the final image to its declared range.

A 49 GB mixed-precision production package exercised this exact metadata
pattern on H200 for text prefill, 512x512 text-to-image, and reference-image
editing: <https://huggingface.co/justinchuby/sensenova-u1.5-8b-mot-onnx-canonical>.

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

## Gemma 4 E2B target and pruned shared-KV assistant

The Gemma 4 E2B pair (examples 23 and 24) exercises a hybrid-attention target and
its cacheless drafter with one generic vocabulary — no model name appears in the
schema. **Both examples are reduced illustrations**: the real target has 35
decoder layers with 15 physical KV owners (3 full-attention at layers 4/9/14,
head_dim 512; 12 sliding-attention, head_dim 256) and 20 shared layers that own
no cache, and the real drafter has 4 layers (3 sliding + 1 full). The YAML slices
below carry a reduced owner set (2 full + 1 sliding owner at illustrative
indices) so the catalogue stays readable; those owner counts and indices are
illustrative, not the model's real graph facts. The borrowed-KV and speculative
CONTRACT is shown exactly as the real graph exposes it.

The **target** (`google/gemma-4-E2B-it`, example 23) is a plain decoder whose
resolved decode ABI reports `kv_ownership: owned`. It declares:

- separate `full_attention` (global, `evictable_prefix: false`) and
  `sliding_attention` (local window 512, `evictable_prefix: true`) groups;
- heterogeneous global/local geometry: the real heterogeneity is head WIDTH — the
  global `full_head_dim` (512) is twice the local `sliding_head_dim` (256). It is
  grouped-query attention with one KV head (`num_key_value_heads: 1`), so key and
  value share one head-count symbol per group. (A model whose K and V head counts
  differ is equally expressible — see example 1 — this checkpoint's do not.)
- fewer physical KV owners than logical layers (`num_kv_shared_layers`): the real
  owners are the 3 full layers 4/9/14 and 12 sliding layers; `layer` orders and
  pairs the physical owners and does not enumerate every logical layer.

This E2B checkpoint is **dense** (`enable_moe_block: false` / `num_experts: null`),
so it declares no MoE metadata — none is invented. The sparse mixture-of-experts
variant is a **different model**, `google/gemma-4-26B-A4B-it` (example 25); MoE is
attributed there, never to E2B. `final_logit_softcapping` and
`tie_word_embeddings` are graph-internal to the E2B decoder artifact and are not
restated as metadata.

The **assistant** (`google/gemma-4-E2B-it-assistant`, example 24) shares one
workflow with the target and owns nothing. Its full/sliding aliases are
`access: read_only` and its carry is folded into the fused input, so its resolved
ABI is `kv_ownership: shared` with no KV transitions and no state pairs. The
speculative contract wires the rest generically:

- single merged borrow: the drafter takes one `shared_kv.full_attention.{key,
  value}` and one `shared_kv.sliding_attention.{key,value}` input — a merged view
  of each group that maps to no specific owner — so each read-only alias names one
  representative owner cell as its anchor and carries NO `layer`;
- `proposal_execution: {kind: chained, token_embedding_input, logits_output,
  folded_carry_output}` — the drafter emits `projected_state` as an output that
  re-enters as the trailing half of the fused `inputs_embeds =
  concat(target_input_embedding(token), carry)`; the first-step carry is
  `port_bindings.target_hidden_context`. The tied target embedding is
  graph-internal, so no external `shared_weights` file is referenced;
- `shared_state: [full_attention, sliding_attention]` names the frozen groups the
  drafter reads;
- the sparse, ordered-embedding LM head routes through centroids
  (`num_centroids`, `centroid_intermediate_top_k`) inside the graph, but the
  drafter still emits the full target vocabulary axis, so the relationship is
  `vocabulary: {kind: identical}`. (A drafter exposing a smaller pruned axis would
  use `subset`; one emitting centroid/cluster ids needing a translation table
  would use `mapped`.) With standard speculative rejection sampling the accepted
  output is the target's, so the drafter is `distribution_preserving: true` even
  with its lossy pruned head;
- `rollback_state` lists every rewound target KV cell; the folded carry has no
  state cell and is recomputed from committed tokens, so it is not rewound.

`crates/onnx-genai-metadata/tests/gemma4_e2b_workflow.rs` resolves both examples
and asserts this ownership, the read-only-versus-read-write split on the shared
groups, the layer-less merged borrow, the folded chained carry (and that it fails
closed and stays backward-compatible with the `recurrent` form), the identical
vocabulary, and the rollback coverage.

## Gemma 4 26B-A4B — the MoE variant (dense E2B is not)

`google/gemma-4-26B-A4B-it` (example 25) is the sparse mixture-of-experts model in
the Gemma 4 family; the E2B examples above are dense and must not be conflated
with it. Every fact below is verified from the pinned config (snapshot
`4d7ae498`) and `modeling_gemma4.py`, not extrapolated from E2B:

- **MoE FFN** — `enable_moe_block: true`, `num_experts: 128`, `top_k_experts: 8`,
  `moe_intermediate_size: 704`, plus an always-on dense **shared** expert
  (`intermediate_size: 2112`). The router scores with `softmax`, selects
  `top_k`, and normalizes the weights. Declared once under
  `model.mixture_of_experts` (128 routed + 1 shared, 8 per token) with legal
  expert-parallel facts under `model.sharding.expert_parallel`.
- **Hybrid attention** — 30 layers, 5:1 local:global (25 sliding + 5 full at
  layers 5/11/17/23/29), sliding window 1024.
- **Heterogeneous global/local geometry in both axes** — the global group has
  `num_global_key_value_heads: 2` at `global_head_dim: 512`; the local group has
  `num_key_value_heads: 8` at `head_dim: 256`. So the two groups differ in KV head
  COUNT and head WIDTH, expressed as independent symbolic axes. Grouped-query
  attention (16 query heads) with `attention_k_eq_v: true`.
- **No shared/pruned KV layers** — `num_kv_shared_layers: 0`, so every layer owns
  its KV. Cross-layer KV sharing is an E2B feature, absent here.

The catalogue slice is a reduced illustration (1 full + 1 sliding owner at real
indices); the real per-layer owner set is stated in the YAML comment.

## Existing evidence

The separate [Qwen Image Edit evidence record](../evidence/qwen-image-edit-2509/README.md)
and [`qwen_image_edit_workflow_e2e.rs`](../../../crates/onnx-genai-engine/tests/qwen_image_edit_workflow_e2e.rs)
are real repository evidence; they do not turn this illustrative catalogue YAML into
an artifact-backed package. ComfyUI import behavior is documented and tested through
[`COMFYUI_IMPORT.md`](../../../docs/genai/COMFYUI_IMPORT.md) and the
`onnx-genai-comfyui-config` crate. No outputs or metrics are claimed here.
