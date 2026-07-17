# Weight Offload and Paging for Huge MoE Models

> **Status:** Approved — Phase 1 cleared to implement (owner @justinchuby, 2026-07-14). Open questions resolved; see §11.
>
> **Primary targets:** GLM-5.2 and DeepSeek-V4-Flash-class sparse MoE models.
>
> **Reference UX:** llama.cpp/Unsloth-style memory mapping and partial GPU offload:
> the same model runs under a small memory budget, while larger machines
> automatically keep more weights resident.
>
> **Date:** 2026-07-14

## 1. Executive recommendation

Treat immutable model weights as a three-tier hierarchy:

```text
read-only external-data mmap        bounded host cache              bounded device cache
Disk / filesystem backing   <---->  RAM (pageable or pinned)  <----> VRAM
        cold                              warm                         hot
```

The runtime should keep one canonical compressed representation on disk, identify
independently addressable weight regions, and lease only the regions needed by the
next operation. Dense/shared tensors receive the highest residency priority. MoE
expert tensors are expert-major, divided into bounded transfer pages, and admitted
by observed routing heat. A lease pins its pages until the CPU kernel returns or the
device completion fence signals.

The recommended implementation is a **parallel, weight-specific residency system**,
not storage of experts in `onnx-genai-kv`. It should reuse or extract the KV
subsystem's generic ideas—tier identities, byte budgets, LRU/priority admission,
promotion, prefetch hints, and pin/lease state—but not its token/sequence keys,
mutable KV payloads, copy-on-write rules, or tensor geometry. Existing design already
states that expert weights are immutable model data, not KV, and calls for a separate
weight API while reusing KV concepts
([DESIGN.md lines 9195-9213](DESIGN.md#L9195-L9213)).

Keep the loader's existing `WeightStore` as the immutable backing catalog. Add a
weight-residency layer beside it, preferably a dedicated runtime crate/module rather
than expanding `onnx-runtime-memory`: that crate is deliberately a pure,
EP-independent activation-liveness planner
([lib.rs lines 1-23](../crates/onnx-runtime-memory/src/lib.rs#L1-L23)). The residency
layer serves a narrow `ExpertStore` facade to fused MoE kernels and a more general
`WeightResidencyManager` to layer placement.

Two fast paths are mandatory:

1. **Fully resident:** if the planned weights fit, allocate/upload once, use stable
   pointers, and avoid eviction or per-token copies.
2. **Paged:** if they do not fit, the same kernels consume bounded leases from mmap,
   host RAM, or VRAM. Capacity degrades into latency rather than model-load failure.

“Any-size machine” still has a physical floor: enough storage for the package,
address space for its mappings, and memory for one activation/scratch set plus one
bounded weight tile. The design removes the requirement that RAM or VRAM hold the
whole model or even a whole active expert.

## 2. Problem framing

### 2.1 Total bytes, active bytes, and resident bytes are different

A dense model reads nearly every parameter for every token, so disk-backed execution
is possible but quickly becomes storage-bandwidth-bound. Sparse MoE changes the
capacity equation:

```text
total model bytes = shared dense bytes + all expert bytes
active token bytes = shared dense bytes + top-k experts per MoE layer
resident bytes     = policy choice constrained by RAM/VRAM budgets
```

GLM-5.2 is the motivating extreme: the quantization design records 744B total
parameters but about 40B active per token, and cites community packages around
223–245 GB for dynamic 1–2-bit variants rather than roughly 1.5 TB uncompressed
([SUB4BIT_QUANT.md lines 8-16](SUB4BIT_QUANT.md#L8-L16)). DeepSeek-V4-Flash is the
second target because it has the same useful systems property: many expert parameters,
but sparse expert activation.

Sparse activation makes offload tractable only if the graph preserves the MoE unit.
A decomposed graph exposes every expert as ordinary initializers; a fused MoE op can
compute routes first, union selected expert IDs across the batch, acquire only those
weight slices, and release them after compute
([SUB4BIT_QUANT.md lines 281-302](SUB4BIT_QUANT.md#L281-L302)).

### 2.2 Two operating regimes

#### Tiny machine: storage-backed execution

- External weights remain in read-only mmap files.
- CPU kernels read compressed blocks directly from mapped expert pages.
- The explicit host cache may be small or zero; clean mmap pages remain reclaimable by
  the OS.
- Shared weights are assigned highest priority but may still be streamed when the RAM
  budget cannot hold them.
- GPU use is optional. If present, one or a few bounded tiles are staged to VRAM.
- Prefill may process active experts in waves so the union of a large batch never has
  to fit simultaneously.

This mode optimizes for bounded owned memory and OS-reclaimable mapped residency, not
high tokens/second.

#### Big machine: resident execution with an offloaded tail

- Shared weights and the hottest/fullest useful layer set are resident.
- If the complete compressed model fits VRAM, paging is disabled after startup.
- Otherwise the planner keeps whole layers and/or hot experts on the GPU, retains the
  remainder in host RAM, and uses disk only as immutable backing.
- Transfers use pinned staging and asynchronous H2D prefetch when profitable.
- On an H200-class system, the default should maximize stable VRAM residency, preserve
  headroom for KV/activations/scratch, and page only the expert tail.

The same package and operator semantics serve both regimes. Placement changes
latency, never routing, quantization format, or numerical policy.

## 3. Current codebase foundations and gaps

### 3.1 Disk foundation: external-data mmap already works

The loader records every external initializer as `(path, offset, length, dtype, dims)`,
maps each backing file read-only, validates each range, and returns borrowed slices
from the live mapping
([weights.rs lines 19-83](../crates/onnx-runtime-loader/src/weights.rs#L19-L83),
[lines 113-167](../crates/onnx-runtime-loader/src/weights.rs#L113-L167)). `WeightRef::External`
already carries the range and shape needed to derive expert subranges
([tensor.rs lines 72-106](../crates/onnx-runtime-ir/src/tensor.rs#L72-L106)).

For a host-accessible EP, executor construction aliases aligned initializer bytes
with a borrowed `DeviceBuffer` rather than allocating and copying them. The comments
explicitly rely on OS demand paging so weights may exceed RAM
([executor.rs lines 691-733](../crates/onnx-runtime-session/src/executor.rs#L691-L733)).
The CPU EP correctly treats borrowed buffers as foreign mmap memory and does not free
them
([provider.rs lines 176-200](../crates/onnx-runtime-ep-cpu/src/provider.rs#L176-L200)).
The EP API defines the same borrowed-buffer ownership and read-only lifetime contract
([provider.rs lines 68-87](../crates/onnx-runtime-ep-api/src/provider.rs#L68-L87),
[lines 121-145](../crates/onnx-runtime-ep-api/src/provider.rs#L121-L145)).

This is the cold-tier foundation. Mapping a file does not mean all bytes are resident;
file-backed pages enter RAM on demand and can be reclaimed cleanly.

### 3.2 The current executor is not a device offloader

The executor still creates one initializer binding for every graph initializer during
build. Host EPs can borrow mmap, but a non-host EP takes the allocate-and-copy branch,
which uploads every initializer eagerly
([executor.rs lines 698-743](../crates/onnx-runtime-session/src/executor.rs#L698-L743)).
It also owns one EP for the whole plan, not per-node CPU/GPU placement
([executor.rs lines 220-236](../crates/onnx-runtime-session/src/executor.rs#L220-L236),
[lines 677-681](../crates/onnx-runtime-session/src/executor.rs#L677-L681)). Therefore
partial GPU offload needs both lazy initializer binding and multi-EP/layer placement;
it cannot be implemented as an allocator tweak.

The transfer API has the right shape but is incomplete for overlap. `ExecutionProvider`
offers `copy_async` and `Fence`
([provider.rs lines 237-241](../crates/onnx-runtime-ep-api/src/provider.rs#L237-L241),
[lines 258-295](../crates/onnx-runtime-ep-api/src/provider.rs#L258-L295)), while the
CUDA implementation currently performs a synchronous copy and returns an already
signalled placeholder fence
([provider.rs lines 220-225](../crates/onnx-runtime-ep-cuda/src/provider.rs#L220-L225)).
True prefetch requires stream-ordered host-to-device copies and awaitable completion.

### 3.3 Existing MoE representation is suitable for slicing, not yet paging

The CPU `com.microsoft::MoE` kernel accepts ORT's expert-major canonical tensors and
validates shapes whose first dimension is `experts`
([moe.rs lines 1-6](../crates/onnx-runtime-ep-cpu/src/kernels/moe.rs#L1-L6),
[lines 155-180](../crates/onnx-runtime-ep-cpu/src/kernels/moe.rs#L155-L180)). Its
per-row execution already indexes contiguous per-expert FC1/FC2 slices
([moe.rs lines 231-317](../crates/onnx-runtime-ep-cpu/src/kernels/moe.rs#L231-L317)).
That layout is the correct storage boundary.

The current correctness kernel nevertheless materializes the complete FC1 and FC2
inputs before routing
([moe.rs lines 225-229](../crates/onnx-runtime-ep-cpu/src/kernels/moe.rs#L225-L229)).
Likewise, `BlockQuantizedMatMul` dequantizes a constant packed matrix into a full f32
`OnceLock<Vec<f32>>`
([block_quantized_matmul.rs lines 77-83](../crates/onnx-runtime-ep-cpu/src/kernels/block_quantized_matmul.rs#L77-L83),
[lines 151-166](../crates/onnx-runtime-ep-cpu/src/kernels/block_quantized_matmul.rs#L151-L166),
[lines 189-212](../crates/onnx-runtime-ep-cpu/src/kernels/block_quantized_matmul.rs#L189-L212)).
Those are valid correctness baselines but must not be used for huge offloaded experts.
The paging fast path needs fused MoE and compressed-domain kernels.

`MOE_SUPPORT.md` already requires expert-major contiguous external data so each expert
slice can be computed from the initializer descriptor without materializing the tensor
([MOE_SUPPORT.md lines 289-302](MOE_SUPPORT.md#L289-L302)). This document adopts that
contract.

### 3.4 KV tiering provides concepts, not a reusable weight implementation

The KV crate names GPU, CPU, and disk tiers
([lib.rs lines 53-69](../crates/onnx-genai-kv/src/lib.rs#L53-L69)) and its page table
tracks page identity, refcount, device, last access, and LRU promotion/demotion
([page_table.rs lines 309-340](../crates/onnx-genai-kv/src/page_table.rs#L309-L340),
[lines 870-925](../crates/onnx-genai-kv/src/page_table.rs#L870-L925)). The paged cache
can promote a requested logical range
([paged_cache.rs lines 509-553](../crates/onnx-genai-kv/src/paged_cache.rs#L509-L553)).
The connector API also contains useful lookup/fetch/prefetch/pin/evict vocabulary
([connector.rs lines 409-458](../crates/onnx-genai-kv/src/connector.rs#L409-L458)).

Direct reuse is unsafe and misleading:

- `PageTable` is sequence/token-oriented and stores mutable KV-specific f32/int8/fp8
  vectors and per-token scales
  ([page_table.rs lines 309-351](../crates/onnx-genai-kv/src/page_table.rs#L309-L351)).
- `KvCacheConnector` keys data by model, token-prefix hash, chunk index, and layer range,
  not immutable file regions
  ([connector.rs lines 65-98](../crates/onnx-genai-kv/src/connector.rs#L65-L98)).
- The shipped hot/cold page movement is currently bookkeeping over host-owned payloads;
  the tier module says both tiers are in host RAM
  ([tiered.rs lines 1-8](../crates/onnx-genai-kv/src/tiered.rs#L1-L8)).
- `LocalTieredConnector` explicitly does not implement real disk spill and retains an
  authoritative owned host payload
  ([local_tiered.rs lines 53-59](../crates/onnx-genai-kv/src/local_tiered.rs#L53-L59),
  [lines 107-123](../crates/onnx-genai-kv/src/local_tiered.rs#L107-L123)).
- `fp8.rs` is a software E4M3FN/E5M2 codec for KV payload compression
  ([fp8.rs lines 1-10](../crates/onnx-genai-kv/src/fp8.rs#L1-L10)); it is not the
  weight-format layer for MXFP4 or IQ blocks.

The right reuse boundary is generic policy primitives after they are factored away
from KV semantics. Weight storage needs immutable external ranges, representation-aware
byte accounting, alignment, I/O and device-copy state, and completion-fenced leases.

## 4. Proposed architecture

### 4.1 Components

```text
ONNX loader / WeightStore
  owns read-only mmaps and validated WeightRef ranges
                 |
                 v
WeightRegionCatalog
  classifies shared tensors and expert subranges; records format/layout/alignment
                 |
                 v
WeightResidencyManager  <---- Resource Governor sub-budgets
  cold mmap | warm host pages | hot device pages | LRU/heat | in-flight state
                 |
         +-------+--------+
         |                |
         v                v
ExpertStore facade    static layer placement
fused MoE kernels     dense/attention/embedding/lm-head bindings
```

Suggested logical interfaces (names are provisional):

```rust
struct WeightRegion {
    id: WeightRegionId,
    backing: ExternalRange,       // path identity + offset + length
    class: WeightClass,           // Shared or Expert { layer, expert, role }
    representation: WeightFormat, // f16, int4, MXFP4, IQ*, ...
    alignment: usize,
    transfer_page_bytes: usize,
}

trait WeightResidencyManager {
    fn lease(&self, request: WeightRequest) -> Result<WeightLease>;
    fn prefetch(&self, request: WeightRequest);
    fn observe_routes(&self, layer: usize, experts: &[u32]);
    fn usage(&self) -> WeightResidencySnapshot;
}

trait ExpertStore {
    fn ensure_resident(
        &self,
        layer: usize,
        experts: &[u32],
        target: WeightTarget,
    ) -> Result<ExpertLease>;
}
```

A lease contains stable mapped, host, or device views plus any readiness fence. Its
lifetime prevents eviction. Device leases remain live until stream completion, not
merely until kernel launch returns.

### 4.2 Expert is the policy unit; page/tile is the capacity unit

A whole expert is convenient for heat and LRU decisions but may itself exceed a tiny
machine's free RAM/VRAM. Store each expert contiguously, then divide its FC1/FC2/FC3,
scale, zero-point, and bias ranges into page-aligned transfer tiles (for example,
tens of MiB, tuned by device and storage).

- **Admission:** choose experts by heat/priority.
- **Transfer:** move bounded pages/tiles.
- **Compute:** consume direct compressed blocks or double-buffered panels.
- **Atomicity:** a logical expert lease groups all companion ranges required by the
  current kernel wave; it does not imply the whole expert is copied at once.

This makes peak owned memory proportional to active tiles and scratch, not model size
or expert size.

### 4.3 Tier semantics

#### Cold: read-only mmap backing

- Canonical bytes are ONNX external data and remain immutable.
- A cold hit returns a checked subrange of the existing mmap.
- CPU direct-compressed kernels may consume that range without a host copy.
- Optional sequential/random access advice and readahead are hints only.
- Clean mapped pages can be discarded after use; strict budget reporting must
  distinguish owned host-cache bytes from OS page-cache/RSS, which is not fully under
  runtime control.

Inline initializers are acceptable for small shared tensors, but offloadable expert
pools must use external data; otherwise the model protobuf itself forces ownership of
all expert bytes.

#### Warm: bounded host RAM

Warm entries are optional derived copies of canonical packed pages:

- pageable aligned pages for CPU reuse;
- pinned pages for repeated H2D transfer;
- optional CPU-prepacked or dequantized panels only when their expanded byte cost is
  charged to the host budget and measured reuse justifies it.

The warm cache uses byte-based LFRU admission with hysteresis, not entry count. A miss
always falls back to mmap, so a zero-byte host cache remains functional. Do not blindly
copy every mmap page into an owned cache; that duplicates the OS page cache without a
benefit.

#### Hot: bounded device VRAM

A device entry is an EP-owned allocation containing either canonical compressed bytes
or an explicitly versioned device-prepacked representation. It is keyed by
`(region, representation, device)` and charged at actual allocated bytes. Eviction is
legal only when no lease or transfer owns the entry. Failed speculative prefetch must
not displace a leased or demonstrably hotter entry.

On a fully resident plan, entries are pinned for the session and the manager collapses
to stable pointer lookup.

### 4.4 Resource Governor is authoritative

The existing scheduler already defines byte/fraction/auto limits for VRAM, host RAM,
and optional disk spill, with defaults of 90% VRAM and 25% host RAM
([governor.rs lines 11-41](../crates/onnx-genai-scheduler/src/governor.rs#L11-L41)).
It also reserves model weights, activations, and runtime overhead before deriving KV
capacity
([governor.rs lines 81-124](../crates/onnx-genai-scheduler/src/governor.rs#L81-L124)).

Extend this into coordinated sub-budgets:

```text
VRAM ceiling = resident shared weights
             + hot expert/device cache
             + KV cache
             + activations and routing scratch
             + EP/runtime overhead

host ceiling = owned host weight cache and pinned staging
             + host KV
             + host activations/scratch
             + non-reclaimable runtime memory
```

Independent KV and weight LRUs must not race for the last bytes. The governor assigns
sub-budgets and can rebalance them with hysteresis. On lowering a live limit: cancel
speculative reservations, evict unleased weight pages, demote KV, reduce batch/scratch,
and return an actionable minimum-working-set error if still impossible.

Current engine wiring uses provisional 8/16/16-GiB capacities and zero fixed
reservations, with a TODO for EP/OS capacity queries
([engine.rs lines 50-89](../crates/onnx-genai-engine/src/engine.rs#L50-L89),
[lines 133-150](../crates/onnx-genai-engine/src/engine.rs#L133-L150)). Auto mode must not
be considered complete until real free/total RAM, filesystem, and device capacity are
reported.

## 5. MoE expert paging and batching

### 5.1 Per-layer execution

For one admitted token batch:

1. Run the model-exact router and compute exact top-k IDs and aggregation weights.
2. Union selected expert IDs across all token rows.
3. Group rows by expert and compute token counts.
4. Ask `ExpertStore` for a residency plan under the current tier budgets.
5. Execute resident experts together; process the remainder in bounded waves/tiles.
6. Scatter and combine with the original aggregation weights.
7. Release CPU leases immediately and device leases after completion fences signal.
8. Record routes, bytes, stalls, and reuse for future admission/prefetch.

The current MoE kernel already notes batch-union expert grouping as its next phase
([moe.rs lines 23-30](../crates/onnx-runtime-ep-cpu/src/kernels/moe.rs#L23-L30)). The
sub-4-bit design specifies the same union/group/acquire/release sequence
([SUB4BIT_QUANT.md lines 281-296](SUB4BIT_QUANT.md#L281-L296)).

### 5.2 Residency policy

- Shared attention, router, normalization, embeddings, and other dense weights have
  higher base priority than routed experts because they are touched predictably.
- Expert admission combines frequency, recency, bytes, measured load cost, and tokens
  served while resident. Use hysteresis to avoid ping-pong.
- Pin only the minimum control/shared set and explicitly requested hot experts; a tiny
  budget must be allowed to stream preferred shared tensors too.
- A page used by an in-flight kernel or transfer is non-evictable.
- Derived dequantized/prepacked entries are disposable and never the sole copy.

### 5.3 Batching interaction

Batching improves compute and I/O reuse when many rows select the same expert, but a
large prefill can activate a broad expert union. Never require that union to fit at
once. Partition it into waves sized by device slots, host staging, and scratch. The
planner should minimize reloads while preserving stable token-row mapping.

Scheduler-level expert affinity is only a tie-breaker after priority/SLA constraints,
because exact routes are not known until the router executes. The primary optimization
belongs inside the fused MoE op
([MOE_SUPPORT.md lines 478-492](MOE_SUPPORT.md#L478-L492)). Rare routes must not wait
indefinitely for a cache-friendly batch.

### 5.4 Prefetch sources

In increasing order of speculation:

1. exact routes for the current layer;
2. exact union for the current admitted batch;
3. recent per-layer route heat;
4. optional router prediction/lookahead for a future layer or decode step.

A prediction reserves bytes before I/O starts. It is dropped when insufficient
headroom exists, when it would evict a hotter entry, or when predicted saved latency
is below transfer cost. Prediction affects performance only; a miss executes the exact
route normally.

## 6. Quantization synergy

Canonical resident and transferred bytes should stay compressed. MXFP4, IQ formats,
and affine int2/int4 reduce all three important quantities:

- disk footprint and storage bandwidth;
- host-cache footprint;
- H2D traffic and VRAM residency.

`BlockQuantizedMatMul` preserves native GGUF blocks in an opaque `uint8` tensor so
external-data slices remain mmap-able
([SUB4BIT_QUANT.md lines 201-225](SUB4BIT_QUANT.md#L201-L225)). Fused expert tensors
should preserve the same expert-major native blocks.

### Dequantize in the kernel — default

Preferred for decode and constrained machines:

- direct IQ/MXFP4/int2/int4 GEMV/GEMM reads compressed pages;
- no full-expert f16/f32 buffer;
- minimum transfer and resident bytes;
- tiles can be released immediately after their dot products complete.

This is required for the “run from mmap” path.

### Dequantize/prepack on load — opt-in derived cache

Potentially useful for a hot expert on a large CPU machine or for a device library that
requires a prepacked layout:

- pays conversion once and reuses the result;
- consumes much more RAM/VRAM;
- must be keyed by format/kernel/device version;
- must be separately budgeted and evictable;
- should be admitted only after observed reuse exceeds a measured threshold.

The current full-f32 constant cache demonstrates the performance idea but also the
memory hazard. Huge-model mode must default away from it.

## 7. Device offload and partial GPU placement

### 7.1 User model: budget first, layer count as an override

The automatic planner should select the largest stable GPU placement that fits after
KV, activation, scratch, and EP headroom. Also expose an explicit llama.cpp-like
control for repeatability:

```text
device_policy = auto | cpu | gpu_layers:<N> | device_bytes:<SIZE>
```

`gpu_layers:N` places complete transformer blocks where possible, avoiding activation
ping-pong at every node. Shared weights for those blocks are resident; their expert
pools use the remaining device cache. The rest execute on CPU from warm/mmap pages.
An advanced expert pin list may be added later, but raw expert-count configuration is
less stable than a byte budget because expert sizes can differ.

### 7.2 Required execution changes

Partial offload requires:

- per-node or per-partition EP placement rather than one EP per executor;
- explicit host/device transfer edges at partition boundaries;
- lazy initializer bindings so a GPU kernel can receive selected expert pages without
  allocating the complete initializer;
- a paging-aware fused MoE kernel contract (`WeightHandle`/`ExpertStore`), or an
  engine-owned fused op that acquires leases before dispatch;
- stream/event lifetime integration so eviction cannot free in-flight device memory.

For the CPU mmap phase, a fused kernel can index selected subranges from the existing
borrowed full initializer. For GPU paging, a `TensorView` of a hypothetical complete
VRAM tensor is not honest; the kernel API needs a lazy weight handle or selected-page
binding.

### 7.3 H200-class path

- Detect real free HBM and reserve conservative headroom.
- Prefer whole-layer stable placement, then fill an expert cache by heat.
- Keep compressed blocks in VRAM unless a measured device kernel requires another
  representation.
- Keep a bounded pinned-host staging ring.
- Prefetch the next exact/predicted expert wave on a transfer stream while the current
  wave computes.
- If all planned weights fit, eagerly load and pin them, disable eviction, and match a
  conventional resident runtime's hot path.

The design must benchmark fully resident performance separately; offload machinery is
not successful if it slows that case materially.

## 8. Configuration and UX

Use the existing `serving.memory.limits` surface as the global authority; it already
accepts byte, fraction, and `auto` values
([config.rs lines 368-408](../crates/onnx-genai-engine/src/config.rs#L368-L408)). Add a
weight policy below it rather than creating an unrelated memory governor:

```yaml
serving:
  memory:
    limits:
      vram_limit: auto
      host_ram_limit: auto
      disk_spill_limit: auto
    weights:
      mode: auto                 # auto | mmap | resident
      device_budget: auto        # sub-budget, capped by vram_limit
      host_budget: auto          # owned/pinned cache, capped by host_ram_limit
      device_policy: auto        # or gpu_layers:48 / device_bytes:120GiB
      prefetch: auto             # off | exact | heat | predictive | auto
```

Environment aliases for command-line deployments:

```text
ONNX_GENAI_WEIGHT_OFFLOAD=1                    # Phase-1 route-first mmap CPU MoE
ONNX_GENAI_WEIGHT_OFFLOAD_HOST_BYTES=<bytes>   # owned Phase-2 warm-cache override
ONNX_GENAI_WEIGHT_BUDGET=auto|<bytes>          # shorthand resident-weight cap
ONNX_GENAI_WEIGHT_DEVICE_BUDGET=auto|<bytes>
ONNX_GENAI_WEIGHT_HOST_BUDGET=auto|<bytes>
ONNX_GENAI_GPU_LAYERS=<N>
ONNX_GENAI_WEIGHT_PREFETCH=off|exact|heat|predictive|auto
```

`ONNX_GENAI_WEIGHT_OFFLOAD` is opt-in in Phases 1 and 2. When set to `1`, pageable
expert-major external QMoE tensors use route-first execution and bypass
full-pool materialization/dequantization caches. Phase 2 inserts a bounded
derived-f32 expert cache between mmap and compute. Its default owned-byte cap is
the Resource Governor's resolved host-RAM sub-budget; the host-bytes environment
variable can lower or override that cap. A zero-byte cap preserves the Phase-1
map-and-dequantize-per-use path. Unset (the default), or for any non-pageable
tensor, execution follows the existing resident QMoE path.

Precedence: explicit API > environment > YAML > auto defaults. Per-tier weight caps
are subordinate to global ceilings and may be reduced by the governor when KV or
scratch needs guaranteed space.

Suggested auto behavior:

1. inventory compressed shared/expert bytes and minimum scratch;
2. query real free/total RAM, VRAM, and filesystem capacity;
3. reserve safety headroom and the minimum KV/activation budget;
4. if all weights fit VRAM, choose fully resident;
5. else if all weights fit RAM, keep a host-resident backing set and maximize stable
   GPU layers/cache;
6. else choose mmap backing, size warm/device caches from remaining headroom, and
   print an estimated bytes/token and likely storage-bound warning.

At startup, print an explainable plan: total/shared/expert bytes, selected tiers,
resident layer count, cache caps, expected minimum working set, and whether async
prefetch is actually supported. Do not claim asynchronous overlap while the active EP
still implements synchronous copies.

## 9. Observability and correctness invariants

Required metrics:

- mapped bytes versus resident RSS;
- owned host bytes and pinned bytes;
- device resident shared/expert bytes;
- hits/misses by tier and layer;
- disk-read and H2D bytes/token;
- page faults, load latency, and compute stall time;
- active experts, unique experts/batch, and tokens/expert;
- evictions, promotions, lease wait time;
- prefetch issued/hit/late/wasted bytes;
- expanded/dequantized derived-cache bytes;
- fully resident fast-path overhead versus baseline.

Correctness invariants:

1. Backing ranges are bounds-checked and immutable for the session lifetime.
2. Residency never changes quantization format or router/aggregation semantics.
3. A leased page cannot be evicted, unmapped, overwritten, or deallocated.
4. Device release occurs after the completion fence.
5. The manager never allocates an unbudgeted full expert/model expansion.
6. Every derived representation is reproducible from canonical backing bytes.
7. A prefetch miss or cancellation cannot change output.
8. Budget failure is reported before OOM with the minimum required working set.

## 10. Phased rollout

### Phase 1 — mmap disk tier and active-expert CPU access

**Ship independently:**

- formalize a `WeightRegionCatalog` over existing `WeightStore`/`WeightRef` ranges;
- require/validate expert-major contiguous external data for paging-capable MoE;
- add a fused CPU MoE path that computes routes first and reads only selected
  compressed expert slices from mmap;
- disable full-expert/full-pool dequant caches in huge-model mode;
- add mapped/RSS/fault/read-byte and active-expert metrics.

**Measure:** run a model/package larger than RAM with bounded owned memory; verify exact
routes/logits against the dense reference; report cold and warmed tokens/s and bytes
read/token. No GPU or explicit host cache is required.

### Phase 2 — bounded host-RAM expert LRU

**Status (2026-07-17): implemented for the fused CPU QMoE path.** The process-wide
warm cache stores immutable `Arc`-backed derived expert entries and charges their
expanded f32 FC1/FC2/FC3 byte size before allocation. Compute leases retain the
`Arc`, so an entry removed from the index cannot be freed during use. Admission
requires repeated use; frequency and recency select victims, while recently hot
entries receive a short policy pin to prevent a rare route from displacing them.
The cache reserves/evicts before fallible dequant allocation and never admits an
entry that would exceed the owned-host cap. Entries larger than the current cap
stream directly from mmap rather than making model loading fail.

The native engine seeds the cache cap from the Resource Governor's resolved
host-RAM limit. `WeightOffloadStats` reports hits, misses, evictions, current and
peak owned bytes, and the active cache budget separately from mmap size and Linux
RSS/page-fault counters. `ONNX_GENAI_WEIGHT_OFFLOAD_HOST_BYTES=0` explicitly
selects the Phase-1 fallback. Device residency and asynchronous prefetch remain
Phase 3 and Phase 4 work respectively.

**Ship independently:**

- add byte-based host pages, leases, LFRU admission, pinning, and hysteresis;
- support zero-byte cache fallback to direct mmap;
- optionally use pinned staging and CPU-prepacked/dequantized derived entries with
  honest expanded-byte accounting;
- integrate host sub-budget reporting with the Resource Governor.

**Measure:** enforce configured owned-host cap; show convergence on a repeated routing
working set; compare direct mmap, pageable cache, and pinned cache hit rate/latency;
verify no cache thrash under rare routes.

### Phase 3 — device cache and partial GPU offload

**Ship independently:**

- add multi-EP/layer placement and explicit transfer boundaries;
- add lazy device initializer/weight-handle binding for fused MoE;
- implement bounded VRAM expert pages and `gpu_layers:N`/byte-budget planning;
- permit CPU execution for non-GPU layers or waves;
- enforce coordinated weight/KV/scratch VRAM sub-budgets.

**Measure:** models larger than VRAM complete without whole-session CPU fallback or
OOM; sweep GPU layer/device cache budgets; report H2D bytes, stalls, tok/s, and peak
VRAM. On a fitting model, fully resident performance must remain near baseline.

### Phase 4 — asynchronous and predictive prefetch

**Ship independently:**

- implement true stream-ordered H2D and awaitable fences;
- double-buffer expert panels;
- add exact-next-wave, heat-based, then opt-in router-predicted prefetch;
- budget reservations and cancel low-value work under pressure.

**Measure:** prefetch hit/late/waste, hidden transfer percentage, p50/p99 token latency,
and throughput across decode and prefill. Predictive mode graduates to default only
when it improves end-to-end performance without increasing memory violations or tail
latency.

## 11. Open questions for owner review

1. **Lazy initializer boundary:** should paging-aware weights enter kernels through a
   new executor `WeightHandle`, through an EP/custom-op context, or through an
   engine-owned fused MoE path? The current all-inputs-are-`TensorView` contract cannot
   honestly represent a partially resident GPU initializer.
   **Resolution:** Use a general executor `WeightHandle` from the start, compatible with
   existing ORT plugin EPs through capability detection. Paging-capable EPs advertise an
   `nxrt` capability flag and receive a lazy `WeightHandle`; stock ORT EPs receive a
   materialized resident-tensor fallback. Paging is opt-in, never a correctness dependency.
2. **ORT integration:** can upstream `QMoE`/plugin EPs lazily access external expert
   slices, or does practical offload require the private `BlockQuantizedMoE` boundary?
   **Resolution:** `pkg.nxrt::BlockQuantizedMoE` is the offload boundary and alone honors
   lazy expert leases, capability-negotiated with a plain `QMoE` fallback. Mobius emits
   `BlockQuantizedMoE` when the `nxrt` capability is present, otherwise `QMoE`; file an
   upstream ORT issue for lazy-external-weight `QMoE`.
3. **Exporter contract:** which metadata is required beyond expert-major shape to bind
   FC1/FC2/FC3, scales, zero points, shared experts, nonuniform expert sizes, and
   format/layout versions without name inference?
   **Resolution:** Use a hybrid contract: numeric bindings (FC1/FC2/FC3, scales,
   zero-points, shared-expert flag, and per-expert sizes) are explicit op inputs or
   attributes, never name-inferred; residency metadata lives in the package manifest; and
   format/layout version is mandatory and explicit, with the loader hard-rejecting a
   mismatch. Residency metadata is a compact model- or layer-group-level layout descriptor
   (stride, tiling, page size, and expert-range formula) referenced by a small region-group
   ID on each op—O(1)–O(layers), not per-expert. Compute concrete byte ranges from
   `WeightStore` offsets plus the descriptor.
4. **Host budget semantics:** do we promise a cap on owned cache bytes only, or a best-
   effort process RSS cap using mmap advice? OS page-cache residency is not strictly
   controllable by the runtime.
   **Resolution:** The cross-platform contract is a hard cap on owned cache bytes.
   RSS-tightening is advisory, off the hot path, and acts only on already-evicted pages so
   it cannot regress performance, behind a `PageAdvisor` trait (`madvise` on POSIX,
   `Offer` + `DiscardVirtualMemory` on Windows, and a no-op fallback).
5. **Partial-GPU policy:** is `gpu_layers:N` required as a stable public compatibility
   knob, or should bytes plus an explainable placement plan be the primary API?
   **Resolution:** Make a byte budget plus an explainable placement plan the primary API.
   Retain `gpu_layers:N` as a compatibility override and report it back in bytes.
6. **Mixed CPU/GPU MoE:** may one fused layer execute some expert waves on CPU and
   others on GPU, or should the first device phase keep each layer on one compute
   device to simplify ordering and numerics?
   **Resolution:** Phase 3 uses a single device per layer; defer intra-layer expert splits
   to a later measured optimization.
7. **Minimum tile size:** what transfer-page/panel sizes best balance NVMe readahead,
   pinned-memory pressure, direct compressed kernels, and GPU occupancy across MXFP4,
   IQ, and affine int2/int4?
   **Resolution:** Default to expert-FC panels comprising whole quant blocks—a tile must
   never split a quant block. Provide a byte-size override that snaps to block boundaries,
   with per-format minimums for MXFP4, IQ, and affine int2–4. Defer auto-tuning to Phase 4.
8. **Governor arbitration:** what minimum KV guarantee and rebalancing hysteresis
   prevent KV/expert-cache oscillation under continuous batching?
   **Resolution:** Use dynamic arbitration: a hard KV floor sized to committed in-flight
   sequences, watermark hysteresis, a minimum rebalance dwell, and admission control at
   batch formation. Thoroughly test oscillation/thrash, KV-floor breaches, and admission
   under continuous batching; these are hard test gates.
9. **Prefetch predictor:** can GLM-5.2/DeepSeek routing be predicted early enough to
   hide storage/H2D latency without duplicating router compute or wasting bandwidth?
   **Resolution:** Use layered opt-in escalation: (a) exact-next-wave by default, (b) a
   heat warm-set, and (c) router prediction as opt-in, graduating to default only when
   measured to help. Provide a trait-based public `ResidencyPolicy` extension point:
   policy advises hints, priorities, and eviction candidates; the Resource Governor remains
   authoritative for budgets, the KV floor, and leases, and cancels low-value work.
   “Policy proposes, Governor disposes”: a bad policy may hurt performance but cannot
   violate memory safety or correctness.
10. **Integrity/lifetime:** should package validation pin file identities/hashes and
    reject replacement/truncation while mmaps and derived cache entries are live?
    **Resolution:** Pin file identity cheaply at load using size plus mtime/inode, or a
    fast header-plus-region-table signature—O(1), with no full re-hash. Offer opt-in
    full hashing for attestation, translate `SIGBUS` to a clean runtime error, and reject
    live truncation or replacement of a mapped package.

## 12. Decision

Proceed, after owner approval, with a **weight-specific parallel residency subsystem**
backed by the loader's existing mmap `WeightStore`. Reuse generalized policy
primitives from `onnx-genai-kv` only after removing KV-specific token, sequence, and
payload assumptions. Make fused MoE the paging boundary, preserve compressed blocks
through all tiers, use leases for pointer lifetime, and let the Resource Governor
coordinate weight, KV, activation, scratch, and EP budgets.
