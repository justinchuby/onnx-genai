# First-Class Mixture-of-Experts Support

**Status:** Research and design only. Every implementation phase in this document is
**NOT YET IMPLEMENTED** unless explicitly described as current behavior.

## 1. Executive recommendation

onnx-genai should treat a sparse Mixture-of-Experts (MoE) block as a first-class
execution and memory-management unit, not merely as a large collection of ordinary
matrix multiplications.

The recommended graph contract is:

1. Keep the **model-specific router as an explicit ONNX subgraph**. It computes two
   logically distinct `[tokens, experts]` tensors: `selection_scores`, including any
   grouped-TopK or bias correction used to choose experts, and
   `aggregation_weights`, containing the model-exact weights used to combine the
   selected expert outputs.
2. Make **`com.microsoft::QMoE` the primary quantized expert op** emitted by Mobius,
   with expert-major packed weights. Use `com.microsoft::MoE` for floating-point
   experts. These existing ONNX Runtime contrib schemas accept the input and router
   probabilities plus stacked expert weights and own top-k dispatch and reduction.
3. Make Mobius able to emit a **dense reference fallback**: explicit TopK/masking,
   per-expert `MatMulNBits` (or `MatMul`), and weighted reduction. It may evaluate
   every expert and is not a performance target; it is the portable correctness
   oracle and fallback when no MoE kernel is available.
4. Do **not** fuse model-specific router policy into the expert op. Router variants
   change faster than expert FFN math, and keeping the router visible preserves exact
   model semantics and makes export failures diagnosable.

The fused op is required for first-class performance and expert streaming: a
decomposed graph exposes hundreds of independent initializers to the generic graph
executor, which cannot naturally union routes, group GEMMs, or make an expert slice
resident only when selected.

## 2. What MoE is and why it matters

An MoE transformer replaces some dense feed-forward network (FFN) blocks with:

- a **router** that scores experts for each token;
- a pool of independent FFN **experts**;
- a sparse top-k choice (often 1, 2, 4, or 8 experts per token); and
- a weighted reduction of the selected expert outputs.

The model can therefore have far more parameters than it activates for one token.
That is the key serving opportunity and the key systems challenge: total expert
weights may exceed VRAM or even RAM, while the active expert working set may fit.

First-class support is needed for the Mixtral, Qwen-MoE, DeepSeek-MoE, and related
model families because a naive ONNX lowering loses the sparse structure:

- it may load all expert weights despite activating only top-k;
- it launches many small GEMMs and gather/scatter kernels;
- it cannot coordinate expert residency with the VRAM ceiling;
- continuous batches may repeatedly load the same expert instead of sharing it; and
- hot-expert skew can create load imbalance across devices.

Strong MoE support is also competitive infrastructure. llama.cpp supports local MoE
execution, while vLLM and SGLang have mature high-throughput MoE paths. The standing
goal is to beat llama.cpp on the experiences we control and approach vLLM-class
serving throughput; sparse models cannot remain an opaque generic-graph workload.

## 3. Colibrì: verified ideas and lessons

This study used Colibrì commit
[`3fd47b7`](https://github.com/JustVugg/colibri/tree/3fd47b7bbd0a8c92fa9344589032d6edc33e40e2).
Colibrì is deliberately specialized for very large MoE inference, especially
GLM-5.2, rather than a general ONNX runtime. Its core systems ideas are nevertheless
directly relevant.

### 3.1 Treat storage as a managed expert hierarchy

Colibrì keeps dense weights resident and streams routed experts from disk. Its README
describes dense int4 weights in RAM, routed experts on disk, a per-layer LRU, an
optional pinned hot store, and the OS page cache as another cache level
([README lines 46-53](https://github.com/JustVugg/colibri/blob/3fd47b7bbd0a8c92fa9344589032d6edc33e40e2/README.md#L46-L53)).
It explicitly treats VRAM, RAM, and storage as one hierarchy without silently
changing precision or router semantics
([README lines 5-10](https://github.com/JustVugg/colibri/blob/3fd47b7bbd0a8c92fa9344589032d6edc33e40e2/README.md#L5-L10)).

Its placement planner separates dense and expert bytes by reading safetensors
metadata, then derives RAM cache capacity and hot/warm/cold expert bytes
([`c/resource_plan.py` lines 37-75](https://github.com/JustVugg/colibri/blob/3fd47b7bbd0a8c92fa9344589032d6edc33e40e2/c/resource_plan.py#L37-L75),
[`c/resource_plan.py` lines 209-257](https://github.com/JustVugg/colibri/blob/3fd47b7bbd0a8c92fa9344589032d6edc33e40e2/c/resource_plan.py#L209-L257)).

**Adopt:** preserve dense/shared weights as a resident set; manage routed experts as
individually pageable, immutable weight regions under one explicit resource budget.
Placement may change latency, but not quantization or routing semantics.

### 3.2 Use routing heat, caching, and lookahead

Colibrì tracks expert usage and supports a pinned hot tier. Its tier policy gives
frequency priority, uses recency as a tie-breaker, and adds hysteresis so experts do
not ping-pong between tiers
([`c/tier.h` lines 6-57](https://github.com/JustVugg/colibri/blob/3fd47b7bbd0a8c92fa9344589032d6edc33e40e2/c/tier.h#L6-L57)).
It also implements asynchronous expert readahead and an experimental router-lookahead
prefetch path
([README lines 65-70](https://github.com/JustVugg/colibri/blob/3fd47b7bbd0a8c92fa9344589032d6edc33e40e2/README.md#L65-L70));
the source uses a dedicated pilot queue/worker for future-layer expert prefetch
([`c/glm.c` lines 2770-2829](https://github.com/JustVugg/colibri/blob/3fd47b7bbd0a8c92fa9344589032d6edc33e40e2/c/glm.c#L2770-L2829)).

**Adopt:** collect per-layer expert heat, use LFRU-like admission with hysteresis, and
prefetch only when the predicted latency saving exceeds transfer and eviction cost.
Prediction is a hint; a miss must never change the output.

### 3.3 Union routes across the batch

Colibrì's batch-union MoE reads each unique expert selected by a prefill/speculative
batch once and applies it to every routed position
([README line 70](https://github.com/JustVugg/colibri/blob/3fd47b7bbd0a8c92fa9344589032d6edc33e40e2/README.md#L70)).
Its MoE forward is explicitly batch-aware and tracks selected expert use
([`c/glm.c` lines 2215-2420](https://github.com/JustVugg/colibri/blob/3fd47b7bbd0a8c92fa9344589032d6edc33e40e2/c/glm.c#L2215-L2420)).

**Adopt:** sort/partition token rows by expert, execute one grouped expert operation
per unique expert (or one grouped launch for many experts), then scatter and combine.
The unit of reuse is the whole active batch, not one request.

### 3.4 Quantize experts without changing routing

Colibrì uses packed int4 experts with per-row scales and dequantization on use
([README lines 63-66](https://github.com/JustVugg/colibri/blob/3fd47b7bbd0a8c92fa9344589032d6edc33e40e2/README.md#L63-L66)).
It also reports a real limitation: cold decode can require about 11 GB of disk reads
per token, so storage bandwidth dominates
([README lines 79-89](https://github.com/JustVugg/colibri/blob/3fd47b7bbd0a8c92fa9344589032d6edc33e40e2/README.md#L79-L89)).

**Adopt:** int4 weight-only quantization for expert capacity and bandwidth. **Do not
adopt** disk streaming as the normal fast path: it is a capacity fallback whose
latency must be visible in planning and metrics.

### 3.5 What is intentionally not copied

- Colibrì's GLM-specific router, attention, MTP, and file naming are not portable
  runtime contracts.
- Topic-affinity routing is useful research, not a correctness dependency.
- Speculative expert loading must not reserve or evict uncontrollably.
- onnx-genai must retain ONNX/Mobius portability and multi-EP execution rather than
  becoming a model-specific engine.

## 4. Current onnx-genai architecture and gaps

There is currently no first-class MoE design in `docs/DESIGN.md`, and
[`docs/OPERATORS.md`](OPERATORS.md) does not list `com.microsoft::MoE` or `QMoE`.
It only mentions MoE as a use for `Concat` and documents ordinary `MatMulNBits`
([`docs/OPERATORS.md` lines 15-84](OPERATORS.md#L15-L84),
[`docs/OPERATORS.md` lines 110-147](OPERATORS.md#L110-L147)).

Relevant existing seams are:

- CPU `MatMul` already has batched/broadcast semantics and a swappable GEMM backend
  ([`matmul.rs` lines 1-17](../crates/onnx-runtime-ep-cpu/src/kernels/matmul.rs#L1-L17)).
  CPU `TopK` and `Gather` provide reference routing primitives
  ([`selection.rs` lines 181-216](../crates/onnx-runtime-ep-cpu/src/kernels/selection.rs#L181-L216),
  [`gather.rs` lines 1-27](../crates/onnx-runtime-ep-cpu/src/kernels/gather.rs#L1-L27)).
- CUDA `MatMul` uses cuBLASLt strided batches
  ([`matmul.rs` lines 1-15](../crates/onnx-runtime-ep-cuda/src/kernels/matmul.rs#L1-L15));
  CUDA attention demonstrates two batched cuBLAS GEMMs around a custom dispatch
  stage on one stream
  ([`attention.rs` lines 21-49](../crates/onnx-runtime-ep-cuda/src/kernels/attention.rs#L21-L49)).
- The in-tree CPU/CUDA EPs do **not currently register `MatMulNBits`, `MoE`, or
  `QMoE` kernels**. Today `MatMulNBits` model execution is an ONNX Runtime/other-EP
  capability documented by the project, not an in-tree grouped-expert kernel.
- The engine owns generation, KV integration, and the Resource Governor
  ([`engine.rs` lines 24-34](../crates/onnx-genai-engine/src/engine.rs#L24-L34),
  [`engine.rs` lines 56-117](../crates/onnx-genai-engine/src/engine.rs#L56-L117)).
- Continuous batching already maintains stable physical rows and a FIFO admission
  queue ([`batched.rs` lines 91-140](../crates/onnx-genai-engine/src/batched.rs#L91-L140)).
- The scheduler owns admission, preemption, priorities, and batch formation
  ([`scheduler/lib.rs` lines 1-8](../crates/onnx-genai-scheduler/src/lib.rs#L1-L8),
  [`scheduler/lib.rs` lines 71-100](../crates/onnx-genai-scheduler/src/lib.rs#L71-L100)).
- The Resource Governor has structures for VRAM, host RAM, optional disk, fixed
  weight/activation/ORT reservations, and an authoritative KV byte budget
  ([`governor.rs` lines 1-5](../crates/onnx-genai-scheduler/src/governor.rs#L1-L5),
  [`governor.rs` lines 73-132](../crates/onnx-genai-scheduler/src/governor.rs#L73-L132)).
  This accounting is **provisional**: the engine currently passes zero bytes for
  model weights, activations, and ORT overhead, and KV disk spill is configuration
  and health reporting only, not an implemented storage tier
  ([`engine.rs` lines 78-88](../crates/onnx-genai-engine/src/engine.rs#L78-L88),
  [`local_tiered.rs` lines 53-59](../crates/onnx-genai-kv/src/local_tiered.rs#L53-L59)).
- `onnx-genai-kv` already has page tables, hot/cold promotion, LRU offload, and
  GPU/CPU/Disk tier concepts
  ([`kv/lib.rs` lines 1-8](../crates/onnx-genai-kv/src/lib.rs#L1-L8),
  [`local_tiered.rs` lines 8-22](../crates/onnx-genai-kv/src/local_tiered.rs#L8-L22)).
  Expert weights are immutable model data, **not KV**, so the concepts should be
  reused but expert storage must not be inserted into the KV APIs.

## 5. Graph and export contract

### 5.1 Primary contract: explicit router + fused expert op

Mobius should emit:

```text
hidden [T,H]
   │
   ├── router subgraph ──> selection_scores [T,E]
   │                       aggregation_weights [T,E]
   │
   └────────────────────────────┐
                               ▼
  com.microsoft::QMoE(hidden, selection_scores,
                     fc1_weights, fc1_scales, [fc1_bias],
                     fc2_weights, fc2_scales, [fc2_bias],
                     [fc3_weights], [fc3_scales], [fc3_bias],
                     [fc1_zero_points], [fc2_zero_points], [fc3_zero_points],
                     aggregation_weights)
      attributes: k, expert_weight_bits=4, block_size,
                  activation_type, normalize_routing_weights,
                  swiglu_fusion
                                │
                                ▼
                         sparse_ffn_output [T,H]
```

This aligns with the existing ORT schemas:

- [`com.microsoft::MoE`](https://github.com/microsoft/onnxruntime/blob/rel-1.27.0/docs/ContribOperators.md#commicrosoftmoe)
  consumes one router tensor and expert-major floating-point FC weights.
- [`com.microsoft::QMoE`](https://github.com/microsoft/onnxruntime/blob/rel-1.27.0/docs/ContribOperators.md#commicrosoftqmoe)
  adds 2/4/8-bit packed expert weights, scales, optional zero points, block size,
  and a separate optional aggregation tensor.

The ORT 1.27 `QMoE` positional input contract is exact and must not be compacted:

```text
 0 input                    11 fc1_zero_points?       15 fc1_global_scale?
 1 router_probs             12 fc2_zero_points?       16 fc2_global_scale?
 2 fc1_experts_weights      13 fc3_zero_points?       17 fc1_act_scale?
 3 fc1_scales?              14 router_weights?        18 fc2_act_scale?
 4 fc1_experts_bias?                                  19 fc1_act_block_scale?
 5 fc2_experts_weights                                20 fc2_act_block_scale?
 6 fc2_scales?
 7 fc2_experts_bias?
 8 fc3_experts_weights?
 9 fc3_scales?
10 fc3_experts_bias?
```

ONNX represents an omitted optional input before a later supplied input with an empty
input name; exporters must preserve those placeholders. For example, an integer
SwiGLU node with no biases or FC3 tensors, but with FC1/FC2 zero points and separate
aggregation weights, uses:

```text
[input, selection_scores, fc1_w, fc1_s, "", fc2_w, fc2_s, "", "", "", "",
 fc1_zp, fc2_zp, "", aggregation_weights]
```

For SwiGLU experts, Mobius should select one canonical layout and record it in graph
attributes. Shared experts remain ordinary dense FFN nodes and are added to the
routed output. Dense-only early layers remain dense.

### 5.2 Why the router remains outside

Mixtral-style softmax top-k, DeepSeek-style sigmoid routing, grouped top-k,
router-bias correction, and model-specific scaling are semantically different.
Mobius should lower those operations faithfully into standard ONNX and keep selection
separate from aggregation:

- pass selection-preserving scores as QMoE input 1, `router_probs`; ORT uses it for
  TopK. For grouped TopK, the explicit router must mask experts outside the chosen
  groups/candidate set so QMoE's global TopK returns the same expert IDs;
- pass `aggregation_weights` as optional input 14, `router_weights`; ORT gathers
  values at the selected expert indices and uses them for reduction. Set
  `normalize_routing_weights` to match the model (normally `0` when the tensor already
  contains final weights); and
- omit input 14 only when the same tensor is correct for both operations.

This mapping represents DeepSeek `noaux_tc`, grouped-TopK, and bias-corrected
selection without applying the selection-only bias to aggregation. `QMoE` owns TopK,
dispatch, expert FFN, and weighted reduction, but not model-specific score formation.

The float `com.microsoft::MoE` schema has no `router_weights`; its single input is
softmaxed and used for both TopK and aggregation. An exact encoding is possible only
when Mobius can construct selection-preserving logits: mask non-selected experts to
`-inf`, put `log(aggregation_weight / row_sum)` at selected indices, and multiply the
MoE output by `row_sum` outside the op if the desired positive weights are not
normalized. If weights are not positive or this transformation cannot preserve the
model contract, float `MoE` cannot represent the router exactly; Mobius must use the
dense reference decomposition (or a future schema with separate IDs/weights).

This split gives:

- exact, inspectable router semantics;
- one expert kernel contract across model families;
- easy router-oracle comparisons;
- freedom to optimize dispatch without changing model math; and
- a natural point for future routing telemetry.

### 5.3 Expert weight layout and external data

Expert tensors must be expert-major and independently addressable:

```text
[expert, output_channel, packed_input_channel]
```

Mobius should store each stacked initializer in contiguous expert slices and use
ONNX external data for large payloads. A future expert store can derive each slice's
file offset and byte length from the initializer's external-data descriptor, shape,
packing, block size, and scale/zero-point tensors. Export validation must reject
layouts whose expert slices cannot be addressed without materializing the entire
tensor.

Proposed model/node metadata (schema names are **speculative and NOT IMPLEMENTED**):

```text
onnx_runtime.moe.version = "1"
onnx_runtime.moe.expert_count = "<E>"
onnx_runtime.moe.top_k = "<K>"
onnx_runtime.moe.weight_layout = "expert_major"

MoE/QMoE node:
  onnx_runtime.layer = "<layer>"
  onnx_runtime.group = "moe_<layer>"
  onnx_runtime.offloadable = "true"
  onnx_runtime.memory.priority = "normal"
```

The existing metadata convention already provides layer, group, offloadability, and
memory-priority hints
([`docs/MODEL_METADATA.md` lines 36-63](MODEL_METADATA.md#L36-L63)).
New keys must be versioned and validated rather than inferred from node names.

### 5.4 Dense reference fallback

Mobius must have a correctness mode that emits an explicit decomposition:

1. compute router probabilities and TopK;
2. form a mask/weight for each expert;
3. execute each expert FFN with `MatMulNBits` or `MatMul`;
4. multiply by routing weights; and
5. sum expert results and add any shared expert.

This may compute all experts. Its purpose is:

- CPU/reference correctness;
- differential testing against `QMoE`;
- support on EPs that lack a fused MoE op; and
- a readable graph for debugging export semantics.

It is not acceptable as the high-performance or streaming representation.

## 6. Kernel design

### 6.1 Common dispatch pipeline

The fused kernel should implement:

1. **Top-k selection** from router probabilities (or accept preselected indices in a
   future schema revision).
2. **Count/prefix-sum** token assignments per expert.
3. **Gather/permutation** of token rows into expert-contiguous segments.
4. **Grouped expert GEMM** for FC1/gate/up, activation, and FC2/down.
5. **Weighted scatter/reduction** back to original token order.

The plan is shaped by `(T, E, K, H, intermediate, quantization, dtype)` and should
special-case decode (`T` small) versus prefill/continuous batches (`T` larger).
Empty experts launch no GEMM.

### 6.2 CPU EP

Integration points:

- `crates/onnx-runtime-ep-cpu/src/kernels/selection.rs`: TopK reference behavior.
- `.../gather.rs` and indexing/movement kernels: correctness building blocks.
- `.../matmul.rs` / `gemm.rs`: dense grouped/batched baseline.
- a future quantized kernel module: expert-batched int4 `QMoE`, using a conversion
  established by the Phase 1 packing-validation gate rather than assuming that
  Mobius/`MatMulNBits` storage is already identical.
- `.../mod.rs`: register `com.microsoft::MoE` and `QMoE`.

The CPU correctness kernel may initially gather into dense scratch buffers and call
existing GEMM paths. The performance kernel should avoid dequantizing a whole expert:
unpack/dequantize blocks inside the dot-product loop and parallelize over active
experts and token rows without oversubscribing Rayon/oneDNN.

### 6.3 CUDA EP

Integration points:

- `crates/onnx-runtime-ep-cuda/src/kernels/matmul.rs`: cuBLASLt batch planning.
- `.../gemm.rs`: shared cuBLASLt mapping and epilogue conventions.
- `.../attention.rs`: precedent for staged batched GEMM plus a custom NVRTC stage
  on one stream.
- `.../mod.rs`: fused-op registration and capability reporting.

The CUDA path needs:

- device TopK/count/prefix-sum;
- token permutation and inverse permutation;
- grouped GEMM (cuBLASLt grouped matmul where suitable, otherwise CUTLASS/custom
  grouped kernels);
- int4 compressed-domain expert GEMM consuming the validated QMoE packing;
- fused activation and routing-weight application where profitable; and
- asynchronous expert H2D copies on a transfer stream with event dependencies.

The kernel must report active-expert counts, tokens/expert, launch count, and scratch
bytes so performance and imbalance are observable.

## 7. Expert residency and streaming

### 7.1 Separate expert store

Create a future `ExpertStore` abstraction in the engine/model-management layer (or a
dedicated weight-storage crate), not in `onnx-genai-kv`:

```rust
trait ExpertStore {
    fn ensure_resident(&self, layer: usize, experts: &[u32], device: Device)
        -> Result<ExpertLease>;
    fn prefetch(&self, layer: usize, experts: &[u32], device: Device);
    fn observe_routes(&self, layer: usize, experts: &[u32]);
}
```

An `ExpertLease` pins immutable expert slices for the duration of a kernel launch.
Eviction cannot invalidate an in-flight pointer. Backing tiers are:

- **hot:** VRAM, GPU compute;
- **warm:** pinned/pageable host RAM, CPU compute or fast H2D;
- **cold:** memory-mapped/read-only model external data on disk.

Use the page-table/tiering lessons from `onnx-genai-kv`, but define weight-specific
page identity, immutable backing, alignment, transfer, and lease semantics.

### 7.2 Resource Governor integration

The VRAM ceiling must cover:

```text
resident dense weights
+ hot expert cache
+ KV cache
+ activations/scratch
+ ORT/EP overhead
<= configured VRAM limit
```

`VramBreakdown` can reserve model weights, activations, and ORT overhead before
deriving KV capacity, but those engine-supplied reservations are currently all zero.
Therefore this design is not yet an enforceable whole-runtime memory ceiling. Phase 3
must first connect real EP/model weight usage, activation/scratch high-water marks,
and ORT/EP allocations to the governor, then add an explicit `hot_expert_bytes`
component and coordinated rebalancing between expert and KV budgets. Lowering a
ceiling must:

1. cancel speculative prefetch reservations;
2. evict unleased coldest experts;
3. demote eligible KV according to the existing priority order;
4. shrink the active batch if scratch/activation pressure remains; and
5. fail clearly if the resident dense set plus minimum scratch cannot fit.

Do not let independent KV and expert LRUs compete blindly for the last VRAM pages.
The governor is authoritative; both managers receive sub-budgets and return usage.

### 7.3 Cache policy and prefetch

Track, per layer and expert:

- frequency and last-use step;
- bytes and current tier;
- load latency by source tier;
- in-flight/pinned/leased state;
- prefetch hit/miss/waste;
- tokens served while resident.

Default admission should be LFRU-like with hysteresis. Prefetch sources, ordered from
least speculative to most speculative:

1. exact routes already computed for the current fused op;
2. union of routes across the admitted batch;
3. recent per-layer heat;
4. predicted next-layer routes.

Prediction must be optional and budgeted. It must not evict a leased expert or a
demonstrably hotter resident expert merely to chase a weak prediction.

## 8. Scheduling, batching, and expert parallelism

### 8.1 Batch formation

The scheduler currently batches by admission, token/KV budget, and priority. Add MoE
affinity only as a **tie-breaker** after SLA/priority constraints:

- prefer requests likely to share resident experts;
- cap waiting time so cache affinity cannot starve rare routes;
- account for expert transfer and scratch bytes in admission;
- preserve stable physical rows used by continuous batching.

Exact routes are unknown until the router runs. Therefore the primary optimization
is inside the fused MoE op: union routes across the already formed token batch.
Cross-request route prediction is optional policy, not a correctness requirement.

### 8.2 Load imbalance

Hot experts may receive far more tokens than others. The execution plan should:

- use token counts to size grouped GEMMs;
- split exceptionally hot experts into multiple work tiles;
- coalesce tiny experts when launch overhead dominates;
- enforce a configurable capacity only if the model contract specifies one;
- never drop or reroute tokens as an implicit overload policy.

Expose per-layer max/mean tokens per active expert and an imbalance ratio.

### 8.3 Expert parallel execution

Single-device is the first target. Multi-GPU expert parallelism later partitions
expert IDs by device and performs all-to-all token dispatch. The placement policy
should favor stable expert ownership and use replication only for measured hot
experts. This requires an explicit distributed design and is not part of Phase 1.

## 9. Quantization

The preferred initial format is per-expert int4 weight-only quantization. Compatibility
between Mobius's existing `MatMulNBits` export and ORT `QMoE` is a **Phase 1
validation requirement**, not an established property. The gate must prove
byte-for-byte weight and zero-point layout, transpose conventions, scale ordering,
and CPU/CUDA prepacking behavior before buffers are reused or converted without a
reference unpack/repack:

- expert-major packed weights;
- explicit block size;
- scales and optional zero points;
- f16/bf16 activations with documented accumulation type;
- no full-expert dequantization buffer.

Mixed precision must be legal per tensor/layer when required for quality. Export and
runtime validation must compare dequantized slices and fused outputs against the
dense reference graph. Quantization may change numerical results within the declared
model format; residency and scheduling must not.

## 10. Attention and KV cache are unchanged

MoE changes the transformer **FFN path only**. Attention semantics, paged attention,
position handling, and KV contents do not change. The existing KV cache remains
authoritative for attention state.

MoE does affect the **shared memory budget** and batch timing: expert residency,
expert scratch, and KV pages compete for VRAM. Reuse the Resource Governor and tiered
memory concepts, not KV tensor formats or KV connector APIs.

## 11. File-level integration map

| Area | Current integration point | Required future work |
|---|---|---|
| Mobius export | External exporter; contracts documented in `docs/OPERATORS.md` and `MODEL_METADATA.md` | Emit explicit router + `MoE`/`QMoE`; expert-major external data; dense reference mode; metadata validation |
| CPU EP | `crates/onnx-runtime-ep-cpu/src/kernels/{matmul,gather,selection}.rs`, `mod.rs` | Reference MoE, then grouped quantized expert kernel and registration |
| CUDA EP | `crates/onnx-runtime-ep-cuda/src/kernels/{matmul,gemm,attention}.rs`, `mod.rs` | Device dispatch/scatter, grouped int4 expert GEMM, async residency |
| Engine | `crates/onnx-genai-engine/src/{engine,batched}.rs` | Expert store lifecycle, leases, telemetry, batch residency plan |
| Scheduler | `crates/onnx-genai-scheduler/src/{lib,policy,governor}.rs` | Expert/scratch byte accounting, affinity tie-breaker, imbalance-aware plans |
| KV concepts | `crates/onnx-genai-kv/src/{page_table,local_tiered,paged_cache}.rs` | Reuse concepts only; do not store experts as KV |
| Documentation | `docs/{DESIGN,OPERATORS,MODEL_METADATA}.md` | Add op coverage and finalized metadata when Phase 1 lands |

## 12. Phased implementation plan

### Phase 1 — representation and dense-fallback correctness

**Status: NOT YET IMPLEMENTED**

Deliverables:

1. Mobius recognizes supported sparse FFN patterns and preserves separate selection
   scores and aggregation weights for the exact router.
2. Mobius emits `com.microsoft::MoE`/`QMoE` with canonical expert-major layout.
3. Mobius can emit the decomposed dense reference graph.
4. `docs/OPERATORS.md` lists MoE/QMoE capability and fallback behavior.
5. Runtime capability checks fail clearly when no fused op or fallback is available.
6. Tiny deterministic fixtures cover top-1/top-2, softmax and sigmoid routers,
   grouped/bias-corrected selection, shared expert, SwiGLU, empty experts, and int4
   scales/zero points.
7. Differential tests compare selection scores, aggregation weights, selected
   IDs/weights, layer outputs, and final logits between source framework, dense
   fallback, and QMoE.
8. Packing tests prove Mobius `MatMulNBits` and ORT `QMoE` layouts byte-for-byte,
   including transposes, zero points, and EP prepacking, or define an explicit
   tested conversion when they differ.

Acceptance gate: a small Mixtral/Qwen-MoE/DeepSeek-style fixture produces equivalent
routes and logits through the dense fallback and fused ORT op. No offload or custom
grouped kernel is required.

### Phase 2 — grouped-expert kernels

**Status: NOT YET IMPLEMENTED**

Deliverables:

1. CPU `MoE` correctness kernel and int4 `QMoE` grouped kernel.
2. CUDA device routing/permutation/scatter and grouped expert GEMM.
3. Decode and prefill plans with bounded scratch allocation.
4. Differential kernel tests against Phase 1.
5. Benchmarks for active experts, tokens/expert distribution, top-k, and batch size.

Acceptance gate: no per-token expert GEMM launch, no full-expert dequantization, and
measured benefit over the dense fallback at representative decode/prefill shapes.

### Phase 3 — expert offload, streaming, and routing-aware scheduling

**Status: NOT YET IMPLEMENTED**

Deliverables:

1. Real EP/model weight, activation/scratch, and ORT overhead accounting wired into
   the Resource Governor; no zero-reservation placeholder accounting.
2. Expert-major external-data paging and `ExpertStore` leases.
3. VRAM/RAM/disk residency under Resource Governor sub-budgets; disk residency
   requires an implemented backing store rather than `DiskTierConfig` alone.
4. Heat-based caching, asynchronous transfer, and budgeted prefetch.
5. Batch-union residency plans and scheduler affinity tie-breakers.
6. Metrics for cache hit rate, transferred bytes, stalls, imbalance, and prefetch
   waste.
7. Stress tests proving the configured VRAM ceiling is not exceeded.

Acceptance gate: models larger than VRAM run without changing routes/precision, hot
working sets converge, and lowering the live ceiling causes deterministic demotion or
an actionable error rather than OOM.

## 13. Open questions and risks

1. **ORT weight ownership:** can an EP/custom `QMoE` kernel lazily access expert
   slices without ORT materializing the full initializer? If not, Phase 3 needs an
   external-data/lazy-initializer extension or an engine-owned custom op.
2. **Schema evolution:** is upstream `QMoE` sufficient for every target router/FFN
   layout, or is a second schema accepting preselected expert IDs and weights needed?
3. **Packing validation result:** Phase 1 must determine whether Mobius
   `MatMulNBits` packing and ORT `QMoE` packing are byte-identical across zero-point
   defaults, transpose conventions, and EP prepacked layouts; otherwise it must
   specify and test the required conversion.
4. **Numerical reproducibility:** grouped/batched kernels may change rounding versus
   per-token execution. Define tolerance and deterministic modes.
5. **Budget arbitration:** static partitioning wastes memory; fully dynamic
   KV-versus-expert competition may oscillate. Start with governor-controlled
   sub-budgets and measured rebalancing.
6. **Cold-tier latency:** SSD streaming is capacity support, not a throughput claim.
   Surface expected bytes/token and avoid prefetch amplification.
7. **Routing prediction:** prediction quality is workload/model dependent and can
   waste bandwidth. It remains opt-in until measured.
8. **Expert parallelism:** multi-GPU all-to-all can dominate small batches and needs a
   separate topology-aware design.
9. **Security and integrity:** external expert slices need the same bounds checking,
   model-signing, and immutable-file assumptions as other model weights.
10. **Model breadth:** Mixtral-like SwiGLU is only the first target. Tests must cover
    shared experts, dense replacement layers, grouped routing, and nonuniform expert
    shapes before claiming broad compatibility.
