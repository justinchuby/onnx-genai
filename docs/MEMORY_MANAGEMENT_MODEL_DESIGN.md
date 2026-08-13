# Unified Memory Management for ONNX Runtime and ONNX Runtime GenAI

**Status:** Design proposal  
**Date:** 2026-08-13  
**Scope:** Process-local device and host memory for inference and generation

## Summary

Today, memory is held independently by ORT execution-provider arenas, model
weights, KV caches, and backend runtime state. Each component manages its own
pool and cannot safely lend unused capacity to, or reclaim capacity from,
another. This strands memory, encourages worst-case reservation, and makes the
configured limit an incomplete view of actual machine pressure.

We want one or more models to use the machine efficiently across heterogeneous
devices: offload when VRAM is insufficient, commit only what the workload uses,
share spare capacity across models and sessions, and preserve headroom for
other applications. A unified authority coordinates these existing memory
mechanisms; it does not replace their specialized allocation and caching
strategies.

The design introduces one shared memory control plane while retaining
specialized allocation and caching data planes.

## Contracts

| ID | Contract | Responsibility |
| --- | --- | --- |
| **C1** | Resource authority | One authority owns the capacity and accounting for a physical resource domain: one per accelerator and one per machine-wide host pool. |
| **C2** | Lease | A holder acquires an RAII lease for a tier, byte count, role, and holder identity before retaining physical memory. Dropping or shrinking the lease returns capacity. |
| **C3** | Allocator and backing | `DeviceAllocator` obtains memory; `VirtualBacking` reserves, maps, and unmaps address space. Neither decides admission or eviction policy. |
| **C4** | Capacity transaction | Growth follows `plan -> reserve -> expose provisional view -> execute -> commit`, with rollback before commit. |
| **C5** | Reclaimable holder | Under pressure, the authority requests bytes from a registered holder. The holder selects safe victims and reports what it actually released. |
| **C6** | Model memory view | A backend exposes either contiguous tensors or cache blocks plus a block table. The view remains valid for the model execution that consumes it. |
| **C7** | Topology and capability | The platform reports capacity, tier aliasing, mapping granularity, transfer paths, and supported model views. Selection is capability-driven. |
| **C8** | Reconfiguration and observability | Authorities expose used, available, oversubscribed, and role-attributed bytes; limits may be lowered only through the reclaim protocol. |
| **C9** | Persistent state bundle | Every model declares all loop-carried state and its lifetime, growth/update pattern, model view, and checkpoint/fork/migrate capabilities. The engine transacts the complete bundle, not only attention KV. |

These contracts must preserve the following invariants:

| ID | Invariant |
| --- | --- |
| **I1 — Single accounting authority** | Every physical byte is charged to exactly one authority. Shared mappings and prefix aliases do not create a second physical charge. |
| **I2 — Charge before commit** | Physical allocation or mapping is preceded by a lease or transactional grant. Already-committed bytes are recorded even when that reveals oversubscription. |
| **I3 — Fail closed** | A denied lease, authority mismatch, or required managed-memory initialization failure never falls back to ungoverned allocation. |
| **I4 — Exclusive ownership state** | Capacity is exactly one of free, transaction-reserved, or committed to a holder. |
| **I5 — Transaction consistency** | A pre-commit failure restores the prior request, cache, and capture state. If components can diverge after commit begins, the engine becomes unhealthy rather than continuing. |
| **I6 — Live-state safety** | The authority never takes memory directly. Leased, pinned, or in-flight data cannot be reclaimed. |
| **I7 — Non-blocking governance** | No thread waits while holding an authority lock, and allocator callbacks do not wait for reclaim. A waiter is woken only after capacity is charged. |
| **I8 — Bytes are authoritative** | Tokens, blocks, and pages are derived from exact model geometry and queried platform granularity; admission includes rounding and transient migration peaks. |
| **I9 — Remap synchronization** | A virtual mapping is not changed while a kernel, transfer, or captured graph may access it. |
| **I10 — State-complete commit** | KV, recurrent state, convolution state, sampler/search state, and request progress commit or roll back at the same logical step. |

## Proposed design

```text
Server / Engine / Scheduler
   plans work in bytes and requests capacity
                    |
                    v
 +----------------------------------------------------------+
 | Memory control plane                                     |
 | DeviceMemoryAuthority (one per physical device)          |
 | HostGovernor          (one per machine; host RAM/disk)    |
 | roles: weights | KV | activation | workspace | overhead  |
 +-------------------------+--------------------------------+
                           | leases / growth grants / pressure
             +-------------+------------------+
             |                                |
             v                                v
 Weight residency holder              KV memory backend
 mmap -> host -> device                +----------------------+
 holder chooses reclaim                | Paged: block tables  |
                                      | VMM: stable flat VA  |
                                      +----------+-----------+
                                                 |
                                                 v
                                DeviceAllocator / VirtualBacking
                                ORT allocator, CUDA VMM, CPU mmap
```

| Design component | Contract and invariant coverage |
| --- | --- |
| Device and host authorities | C1, C2, C5, C8; I1, I2, I3, I6, I7 |
| Scheduler and admission | C4, C7, C8; I4, I5, I8 |
| Weight residency manager | C2, C5; I2, I6 |
| Paged KV backend | C2, C4, C6; I2, I4, I5 |
| Contiguous-VA VMM backend | C2, C3, C4, C6, C7; I1-I5, I8, I9 |
| ORT allocator and I/O-binding adapters | C2, C3, C6, C8; I2, I3 |
| Persistent state manager | C2, C4, C6, C9; I4-I6, I10 |

**Control plane.** `DeviceMemoryAuthority` owns the ledger and stable
`MemoryAuthorityId` for one physical-device compatibility domain. Server
engines sharing a device share this authority. `HostGovernor` is machine-wide
because offload, pinned staging, and disk are shared by every device. Its
ticketed protocol charges a grant before waking a requester.

**Data plane.** `DeviceAllocator` answers *where bytes come from*;
`MemoryGovernor` answers *whether they may be retained*. `VirtualBacking`
implements reserve/map/unmap, while `VirtualBuffer` keeps a stable base address
and leases committed granules. A paged KV backend reserves blocks and publishes
a block table. Both use the same capacity transaction.

**ORT integration.** ORT-owned allocations are made visible by registering a
`GovernedAllocator` and enabling `session.use_env_allocators`; refusal returns
allocation failure rather than escaping the budget. Runtime-owned KV uses
external `OrtValue`s and I/O binding: a flat tensor binds stable VMM addresses;
a paged model binds its cache pool and block table. ORT's arena and planner
continue to optimize transient reuse.

### Required ORT and ORT GenAI changes

| Surface | Proposed change |
| --- | --- |
| `OrtEnv` / process runtime | Registry keyed by physical adapter/topology identity so sessions and EPs share C1. |
| Session / EP creation | Inject the authority before arenas, initializers, or state allocate; late adoption only records accomplished allocations. |
| ORT arenas | Bulk-lease regions; report in-use, cached/reclaimable, pinned, and opaque bytes; add soft limits and `release_to(target)`. |
| Planning | Report persistent and bounded activation/workspace peaks for model and request admission. |
| Persistent state | Replace KV-specific ownership with C9 `StateBundle`, using external `OrtValue`/I/O Binding or the same EP-private contract. |
| Weights | EP-visible stable slots with `ensure_resident` and reclaim; I/O Binding alone cannot page internal initializers. |
| Graph capture | Bind capture to addresses, shapes, view kind, and mapping generation; remap/page-in only outside replay after fences. |
| Windows | Feed DXGI local/non-local budgets and change events into C7/C8; `gpu_mem_limit` remains only an arena limit. |

The minimal integration can live above ORT using registered allocators and
external state tensors. The durable design requires EPs to report auxiliary
memory and participate in reclaim; otherwise the authority must reserve
conservative unattributed headroom.

### State and attention mechanisms

**VMM removes paging from a flat kernel's addressing contract; it does not
eliminate paging-aware kernels.** Dense MHA/GQA can keep flat pointers.
PagedAttention remains useful for block-table exports, finer allocation
quantums, or platforms without remapping. Windowed, sparse, compressed/latent,
quantized, and recurrent kernels still implement their semantic indexing.

| State form | Growth/update | Required model view and state operations |
| --- | --- | --- |
| Dense KV | Append | Flat VA or blocks/table; truncate, fork, share, migrate. |
| Windowed/sparse/compressed/latent KV | Windowed/indexed append | Kernel metadata; reclaim only unreachable ranges. |
| Linear attention / retention | Fixed recurrent summary | Checkpoint, clone, restore, migrate; no token paging. |
| SSM + causal convolution | Fixed SSM state + convolution ring | Atomic update and prefix snapshot of both. |
| Hybrid layers | Growing KV + fixed recurrent state | One C9 bundle and C4 transaction. |
| Speculative/MTP branches | Tentative forked state | COW fork, branch commit/abort, recompute. |

The descriptor declares lifetime, extent, view, mutation, and supported
checkpoint/fork/restore/migrate/recompute operations; it is not a list of model
names.

For capture and weight offload, stable VA preserves pointers, but shapes,
strides, launch geometry, block-table buffers, and all C9 views must also remain
compatible. Remap/page-in occurs outside replay. Paged attention is capturable
with a stable pool/table buffer and bucketed table shapes.

### Capacity transaction

Paged and VMM-backed KV use the same step protocol:

```text
plan -> reserve bytes and pages -> expose provisional model view -> execute
     -> commit all cooperating state
     \-> on recoverable failure: rollback state and release reservation
```

For VMM growth, `prepare_mapped_growth` may transfer allowance from a registered
reclaimable weight holder before any new granule is mapped. For paged attention,
the reservation owns blocks before they appear in a committed request table.
If collaborators disagree after commit begins, the engine fails terminally
rather than continue with divergent cache and request state.

### Extensibility and backend selection

Implementations extend the contracts, not the engine: custom
`MemoryGovernor`/`LeaseAccounting` policy, server-owned
`MemoryAuthorityProvider`, platform `DeviceAllocator`/`VirtualBacking`,
device-specific `KvPageStoreFactory`, and holder-specific reclaim policy all
remain replaceable.

Backend selection must be capability-driven, not model-name-driven:

1. Use paged attention when the model declares block-table inputs and the EP
   implements the operator.
2. Use contiguous-VA VMM when the model expects flat KV tensors and the EP can
   reserve/map device virtual memory.
3. Use the existing static/shared-buffer path when neither contract is
   available.

Granularity and topology are capabilities too. A 2 MiB CUDA granule can be
efficient for large, low-concurrency contexts but wasteful for many short
sequences; small block-table pages may win there. Unified-memory systems must
report that host and device tiers alias the same physical pool so two governors
do not double-admit it.

### Topology scenarios

| Scenario | Authority layout | Expected behavior |
| --- | --- | --- |
| **CPU only** | One host authority covers weights, KV, ORT arenas, and workspace; mmap/disk is colder backing. | Reserve KV and peak execution bytes. Reclaim allocator caches and derived/prepacked weights before live KV; deny work rather than force system paging. |
| **CPU + discrete NVIDIA GPU + Intel NPU** | GPU and NPU each have a device authority and share one host authority for staging/offload. C7 reports PCIe paths, model views, and whether NPU memory is local or shared. | Place partitions first, then lease each resource. Demotion reserves host capacity before transfer. State stays fixed on an EP that cannot remap or migrate it. |
| **CPU + GPU with unified memory** | Host and GPU views alias one physical authority; their limits are not additive. Track wired/resident and pageable/reclaimable bytes separately. | CPU/GPU movement is a residency change, not a second allocation. Admission protects machine headroom; reclaim respects wired GPU work and bandwidth. Apple Silicon/Metal is the primary example. |

These scenarios are selected from C7. The upper layers use the same leases and
transactions; only the authority graph, allocation mechanism, and legal
transitions change.

### Prefix caching, continuous batching, and model switching

| Behavior | What the design enables | Additional requirement |
| --- | --- | --- |
| **Prefix caching** | A committed prefix is an immutable C9 bundle. Blocks/VMM granules share physical backing; recurrent/conv state snapshots at the same token boundary. | Key by model, tokenizer, adapter, positions, dtype/layout, and state schema. Hybrid models reuse only complete bundles; divergence uses COW. |
| **Continuous batching** | Requests lease state independently; each step reserves all selected state deltas and transient peaks before execution. | Variable packing, byte admission/fairness, I10 rollback, and capture buckets covering every view shape. |
| **Local model switching** | Demote/discard reconstructible weights, graph pools, and prefixes, then lend capacity to another model without destroying the process. | Model leases, hysteresis/load-cost policy, stable weight slots, and protection for active request state. |

Evidence is uneven: VMM prefix sharing and native recurrent-prefix restoration
have parity/accounting tests, and paged KV has transactional batching. ORT
hybrid-prefix restore correctly falls back to recompute because full C9 restore
is not wired. Stable weight slots and offload exist, but cross-model residency
policy and model-switch latency gates remain design work.

This turns model switching from session destruction/recreation into residency
management. It also makes the failure mode explicit: if two active working sets
cannot coexist and neither holder can reclaim, the new admission waits or fails
instead of causing an unpredictable device OOM.

### WDDM implications

WDDM is **virtualized discrete memory, not Apple-style unified physical
memory**. GPU virtual addresses remain stable while VidMm may back them with
local VRAM or system memory. DXGI local/non-local budgets change with system
pressure; `SharedSystemMemory` is a maximum, not free capacity.

Treat those budgets as external ceilings: preserve headroom, react to budget
events, and track non-local residency, host RAM, and pinned memory separately.
WDDM is address-stable but placement and latency are opaque. The preferred
policy is a governed resident hot set plus host-mapped cold, single-touch
weights—not copy-map-evict churn. Managed no-spill CUDA VMM pools remain hard
bounds and cannot assume WDDM spill. CUDA managed memory is distinct and
reports limited support on Windows, so all behavior is capability-queried.

Cross-process coordination, cluster placement, and universal disk spill are
out of scope. The first production target is one process, one or more devices,
truthful limits, deterministic rollback, and no ungoverned escape path.

## Related work

| Reference | Relevant precedent | Design implication |
| --- | --- | --- |
| [**vLLM**](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/block_pool.py) | Per-engine fixed-block KV pool with sharing and eviction; its utilization limit is per instance. | Keep block ownership below the shared authority. |
| [**TensorRT-LLM**](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/kvcache.md) | Paged KV, reuse, and host offload; geometry-specific pools are statically split. | Let specialized pools borrow and return C2 leases. |
| [**llama.cpp**](https://github.com/ggml-org/llama.cpp/blob/master/common/fit.cpp) | `--fit` estimates model, KV, and compute memory across heterogeneous backends before load. | Use whole-model fit as admission input, then maintain runtime leases. |
| [**Hugging Face Accelerate**](https://huggingface.co/docs/accelerate/concept_guides/big_model_inference) | `device_map` places weights on GPU, CPU, or disk under static limits. | Validate placement proposals against runtime KV and execution leases. |
| [**MLX**](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html) | CPU and GPU arrays share Apple unified memory under a process allocator. | Use one physical authority with residency classes. |
| [**vAttention**](https://arxiv.org/abs/2405.04437) | Reserves contiguous KV VA and maps physical pages on demand, preserving flat kernel pointers. | Implements C3/C6 alongside, not above, governance. |
| **PyTorch / ORT allocators** | Process/session pools and external allocator hooks; ORT's `gpu_mem_limit` covers only its EP arena. | Feed allocator statistics into C8; account or reserve for the remainder. |

No surveyed stack provides the complete target: a machine-aware authority that
coordinates weights, KV, execution memory, heterogeneous devices, multiple
models, and host headroom. The proposal composes their proven data-plane ideas
under explicit cross-component contracts.

## Alternatives considered

| Alternative | Why not the primary design |
| --- | --- |
| Independent fixed budgets for ORT, weights, and KV | Simple and predictable, but strands capacity and requires the user to guess the right split before load. One model can fail while another pool retains unused bytes. |
| One global allocator | Conflates policy with mechanism. Activations, KV, weights, shared prefixes, and virtual mappings have different lifetimes and ownership rules; some ORT and EP allocations also occur behind specialized arenas. |
| Paged attention only | Efficient for high concurrency, but requires exported block-table inputs and a compatible attention operator. It cannot transparently serve existing ORT graphs that declare flat past/present tensors. |
| Contiguous-VA VMM only | Preserves existing tensor contracts, but coarse device granularity can waste memory for many short sequences, and not every platform exposes equivalent remapping. |
| Rely on OS or driver oversubscription | Can make an oversized model run, especially under WDDM, but placement, eviction, and latency are opaque and the runtime cannot reserve headroom for other applications. |

Paged attention and VMM are therefore complementary data planes selected by C7,
not competing control planes.

## Risks

Confidence below is confidence in the **mitigation**, not confidence that the
risk exists. Evidence refers to the current repository implementation, tests,
and hardware measurements summarized in `MEMORY_ARCHITECTURE.md`.

| Risk | Confidence | Evidence and remaining work |
| --- | --- | --- |
| **Incomplete accounting** | **Medium** | Governed ORT allocation and CUDA weight/KV/workspace leases share one ledger; activation/overhead and ORT VMM KV remain incomplete. Reconcile telemetry and reserve unattributed headroom. |
| **VMM granularity waste** | **Medium-high** | Tests measure a 2 MiB CUDA granule and layout crossovers; token-major cut one floor by 768x. Automatic cross-device selection remains. |
| **Unsafe remap/commit** | **Medium end to end** | Same-VA graph replay, stable weight slots, prefix multi-map/refcounts, and mapped-growth transactions pass GPU tests. In-flight unmap and multi-model/multi-stream stress remain unsupported. |
| **Offload thrash/corruption** | **Low-medium** | Cyclic LRU measured 0% hits versus 74.18% for a stable set; WDDM cold reads beat managed churn. Dynamic stable-slot re-admission has unresolved corruption. Ship static hybrid first; gate dynamic policy on token identity, bytes/token, and tail latency. |
| **Host pressure** | **Medium-low** | Ticketed `HostGovernor` has TLA/refinement and conformance tests; RSS, pinned, disk, and WDDM non-local pressure are not yet one physical authority. |
| **Topology errors** | **Low-medium** | CUDA/WDDM are measured; Intel NPU and true UMA graphs need capability/admission conformance. |
| **External consumers** | **Low** | A process-local authority cannot reclaim opaque or other-process allocations; telemetry and reserve provide backpressure, not a hard guarantee. |

## References

- [`MEMORY_ARCHITECTURE.md`](./MEMORY_ARCHITECTURE.md)
- [`PRESSURE_PROTOCOL_IMPL.md`](./PRESSURE_PROTOCOL_IMPL.md)
- [`native-ort-kv-capacity.md`](./native-ort-kv-capacity.md)
- [ORT memory consumption](https://onnxruntime.ai/docs/performance/tune-performance/memory.html)
- [ORT C API shared allocators and arena shrinkage](https://onnxruntime.ai/docs/get-started/with-c.html#features)
- [ORT CUDA EP options](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
- [ORT device tensors](https://onnxruntime.ai/docs/performance/device-tensor.html)
- [GenAI past/present shared buffer](https://onnxruntime.ai/docs/genai/howto/past-present-share-buffer.html)
- [GenAI paged-attention engine](https://github.com/microsoft/onnxruntime-genai/blob/main/docs/paged_attention_engine.md)
- [WDDM 2.0 GPU virtual memory](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/gpu-virtual-memory-in-wddm-2-0)
- [DXGI video-memory budgets](https://learn.microsoft.com/en-us/windows/win32/api/dxgi1_4/ns-dxgi1_4-dxgi_query_video_memory_info)
- [CUDA Unified Memory paradigms](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#unified-memory)
- [Transformers Mamba state](https://huggingface.co/docs/transformers/model_doc/mamba)
