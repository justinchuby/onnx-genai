# Unified Memory Management: Supporting Detail

**Status:** Supporting appendix for
[`MEMORY_MANAGEMENT_MODEL_DESIGN.md`](MEMORY_MANAGEMENT_MODEL_DESIGN.md)  
**Date:** 2026-08-13

This appendix carries protocol detail, topology examples, evidence, and related
work. Decisions and normative contracts live in the main design.

## A. Definitions

| Term | Meaning |
|---|---|
| **Physical pool** | A capacity that can be independently exhausted: one GPU's private VRAM, host RAM, or a disk tier. On UMA, CPU/GPU views may name the same pool. |
| **Authority** | The canonical accounting identity and ledger for one physical pool within one accounting scope. |
| **Governor** | A lease-granting and pressure interface. One governor may front several authorities/tiers, but each grant names the exact authority. |
| **Accounting scope** | One ORT process and its local quota, optionally delegated by a Foundry serving coordinator. |
| **Reservation ID** | Identity of provisional capacity held before physical backing is committed. |
| **Physical allocation ID** | Process-unique, never-reused identity assigned when backing is committed. Address reuse never reuses this ID. |
| **Mapping generation** | Monotonic version of the mappings/views over one physical allocation. Fences and capture compatibility use `(allocation ID, mapping generation)`. |
| **Lease** | RAII ownership of charged physical bytes by `(authority, allocation ID, tier, bytes, role, holder)`. |
| **Allowance** | Non-owning permission to map/use already-charged physical capacity. It is accounted separately from physical ownership. |

Authority IDs are process-local. Foundry multi-process coordination delegates
quotas; it does not create cross-process physical allocation identity. Shared
CUDA IPC/VMM backing and cross-process weight dedup require a separate global
backing identity and are future work.

## B. Holder and pressure protocol

```mermaid
sequenceDiagram
    participant H as Holder
    participant G as MemoryGovernor
    participant A as DeviceAllocator / VirtualBacking
    participant S as InferenceSession / EP

    H->>G: reserve(authority, tier, bytes, role, holder)
    alt granted
        G-->>H: grant / lease
        H->>A: allocate or map using grant
        A-->>H: allocation ID + stable view
        H->>S: bind/publish model view
        Note over H,S: Holder keeps lease while allocation/view is live
    else pending or denied
        G-->>H: WouldBlock / Denied
        Note over H,G: Allocator fails fast; scheduler may await pressure ticket
    end
    opt later pressure
        G->>H: target bytes, priority, deadline, generation
        H->>A: fence and release a safe victim
        H-->>G: shrink/drop lease and report released bytes
    end
```

Allocator callbacks never wait. A wait-capable scheduler may await a pressure
ticket before opening a transaction. Cancellation, timeout, and ticket drop
return any granted capacity exactly once.

### Multi-authority transaction

Cross-tier/device transactions follow two phases:

1. **Pre-pressure:** discover required source, destination, communication, and
   transient peaks. Negotiate any Foundry delegated quota and perform all
   waiting/reclaim here, while no transaction owns partial reservations.
2. **Try-only reservation:** sort authorities by stable identity and try-reserve
   all of them in that order. A failure releases every reservation and returns
   retry/defer; it never waits while holding another authority's reservation.

The failed transaction retains its original submit identity, priority, and
bounded aging as a non-owning intent at each authority. This prevents fresh
arena/holder requests from barging indefinitely without reintroducing
hold-and-wait. Coordinator-to-worker reclaim is a deadline-bounded pressure
ticket; a worker never waits on it from inside an open transaction. When a
transaction defers, the worker returns delegated quota above its committed
local leases. The serving coordinator retains the worker's submit identity,
priority, and bounded aging as a non-owning intent and reuses it on retry.

Only after all grants exist may holders expose provisional model views and
execute. Pre-commit failure restores state and releases grants. A failure after
commit starts poisons the smallest rebuildable unit:

- state/model-view disagreement poisons that generation engine;
- allocator/authority ledger corruption poisons the worker process;
- sibling engines survive only when their shared authority invariants remain
  independently valid.

An unhealthy engine releases its leases or forces worker termination. Heartbeat
or epoch failure fences future claims and initiates termination, but the
quarantined quota remains charged. Foundry reclaims it only after observing
process exit.

### Completion-feasible admission

Admission must preserve progress, not merely fit the next step:

- each accepted request retains enough guaranteed capacity to reach its next
  releasable checkpoint or configured completion bound;
- waiting requests hold no scarce state;
- multi-model admission never leaves all resident models unable to reach a
  release point.

This is the implementation obligation represented by
[`KvAdmission.tla`](../../specs/tla/KvAdmission.tla) and
[`CoResidency.tla`](../../specs/tla/CoResidency.tla).

## C. Provider bootstrap, lifetime, and device loss

Bootstrap ordering is:

```text
topology discovery
  -> EP/device provider creates allocator/backing
  -> ProcessMemoryManager registers provider, capabilities, and shared handles
  -> OrtEnv registers governed adapters
  -> InferenceSession and holders may allocate
```

ORT owns the allocator ABI. The EP, host backend, or embedder creates the
implementation. `ProcessMemoryManager` pins the provider/context while any
allocator handle, lease, or view is live. Deregistration succeeds only after
holders and sessions quiesce.

Device removal/TDR increments the authority generation and transitions every
affected lease/view to a terminal invalid state. New admission stops
immediately. The initial implementation terminates the affected worker; epoch
fencing blocks new claims, and only observed process exit lets Foundry reclaim
the delegated quota. In-process device recreation is future work.

ORT has two distinct allocator integration paths:

- A custom environment-registered `GovernedAllocator` is the allocation path;
  ORT does not wrap it in BFC. The measured host implementation can therefore
  use a header-contained per-allocation lease, while a device implementation
  must provide its own arena/VMM suballocation.
- Teaching ORT's own BFC arena to bulk-lease and `release_to(target)` is a
  separate upstream ORT change.

`session.use_env_allocators` is required for sessions to use the registered
path. Today `RegisteredAllocator` conservatively leaks its allocation object
because ORT does not expose the last session user; process-manager ownership is
the proposed safe lifetime.

## D. Component responsibilities

| Component | Responsibility |
|---|---|
| OS/driver | Ultimate machine arbitration, including unrelated programs; publishes changing budgets and device-loss events. |
| ClusterCoordinator | Node/model placement and coarse node quotas; never page/token allocation. |
| Foundry `ServingMemoryCoordinator` | Authenticated cooperative quotas across spawned/authorized workers; heartbeat/epoch fencing and process-exit reclamation. |
| `ProcessMemoryManager` | Enforces one worker's local/delegated quotas; owns resource, provider, allocator, and authority registries. |
| `TopologyProvider` | Physical/aliased pools, capacities, mapping granularity, and transfer links. |
| `MemoryAuthorityRegistry` | Canonical authority per physical pool in this process scope. |
| Process `HostGovernor` | Host/disk child quotas and ticketed pressure across sessions/devices in the process. |
| Process `DeviceMemoryAuthority` | One device child quota, mapped allowances, and growth. |
| `CapacityTransactionCoordinator` | Ordered, try-only reservation across authorities. |
| ORT GenAI engine | Generation semantics, state transactions, admission, and holder policy. |
| Model residency holder | Hot/warm/cold weights and stable slots; chooses reclaim victims. |
| `StateBundle` / `KvPageStore` | KV/recurrent/conv state, prefix sharing, fork/checkpoint/migrate semantics. |
| Communication holder | Collective/IPC staging and readiness fences; registers required lease set before enqueue. |
| ORT/EP arena | Local suballocation from bulk leases; returns wholly reclaimable regions. |
| `DeviceAllocator` / `VirtualBacking` | Mechanism only: allocate/free or reserve/map/unmap. |
| `InferenceSession` / EP | Graph plan/bind/run, transient-peak reporting, kernels, and capture. |

## E. State mechanisms and model views

| State form | Growth/update | Required operations |
|---|---|---|
| Dense KV | Append | Flat VA or blocks/table; truncate, fork, share, migrate. |
| Windowed/sparse/compressed/latent KV | Windowed/indexed append | Kernel metadata; reclaim only unreachable ranges. |
| Linear attention / retention | Fixed recurrent summary | Checkpoint, clone, restore, migrate; no token paging. |
| SSM + causal convolution | Fixed SSM state + convolution ring | Atomic update and prefix snapshot of both. |
| Hybrid layers | Growing KV + fixed recurrent state | One state bundle and transaction. |
| Speculative/MTP branches | Tentative forked state | COW fork, branch commit/abort, recompute. |

A model capability descriptor declares lifetime, extent, view, mutation, and
checkpoint/fork/restore/migrate/recompute support. It is not a list of model
names.

### VMM versus PagedAttention

VMM separates virtual address from physical commitment. Existing flat
MHA/GQA kernels can remain unchanged while the runtime commits pages on demand.
This is valuable for current ORT graphs, broad kernel compatibility, stable
addresses, graph capture, and low-concurrency local inference.

PagedAttention keeps block placement visible to the kernel. It remains useful
when:

- the exported model already declares block-table inputs;
- many short/variable requests make the VMM granule too coarse;
- fine-grained prefix COW/block ownership matters;
- the platform cannot remap virtual memory.

If neither VMM nor a block-table model/EP is available, the runtime uses static
contiguous mode. It reserves and charges worst-case bytes at admission, so the
bound remains hard; pressure-time reclaim and elastic lending are unavailable.

The mechanisms are composable: a paged block pool may itself allocate from VMM.
The approximate allocation quanta are:

```text
VMM quantum   = mapping granule x contiguous fragment count
Paged quantum = block tokens x KV bytes per token
```

The repository's current decision remains flat VMM for native CUDA because its
models/kernels do not expose block tables. Paged attention is an optional
capability for compatible exported models/EPs, not the default replacement.
Captured paged attention with bucketed table shapes is a design assumption that
still needs hardware validation in this repository.

### Graph capture and weight offload

Stable VA preserves pointers, but capture also requires compatible shapes,
strides, launch geometry, table buffers, and state views. Mapping/page-in occurs
outside replay after fences. Host-mapped device pointers are capture-compatible
and replay bit-identically (#877/#880/#912) — capture is not the constraint on
zero-copy tiering; aggregate host-mapped read capacity is.

Dynamic weight eviction/re-admission is **not shippable today**: repository
measurements found output corruption for large stable-slot residents. Until the
root cause is fixed and token identity is gated, the supported policy is a
static hot set; lending is limited to capacity never admitted into that set.

## F. Topology and coordination scenarios

| Scenario | Authority shape |
|---|---|
| CPU only | One process host authority; mmap/disk is cold backing. |
| Discrete GPU + NPU | Separate child authority per private pool; one shared host quota for staging/offload. |
| UMA | Host/device views resolve to one aliased authority fronted by one governor. Device-wired, host-pageable, and shared-coherent are residency classes, not additive pools. |
| One GPU, multiple Foundry workers | Serving coordinator delegates cooperative child quotas; raw CUDA contexts/allocators remain per process. |
| Multi-GPU node | One quota per private GPU pool plus shared host/communication peaks; transaction reserves every participant before commit. |
| Multi-node | Cluster coordinator assigns placement/coarse node quota; all page/step work stays local. |

External programs are not holders. Foundry computes an effective serving budget
from operator limits, OS/driver signals, and safety reserve, then stops admission
or requests cooperative reclaim as headroom changes.

For a participating worker:

```text
effective worker budget = min(delegated quota, local OS-derived ceiling)
```

Normal serving-policy reclaim originates at the coordinator. The worker treats
its local OS signal as a safety floor: it stops admission and notifies the
coordinator, initiating local emergency reclaim only when needed to avoid an
imminent allocation failure.

### WDDM

WDDM is virtualized discrete memory, not Apple-style UMA. GPU virtual addresses
remain stable while VidMm may back them with local or system memory. DXGI
local/non-local budgets are changing external ceilings; `SharedSystemMemory` is
a maximum, not free capacity.

A host-backed GPU allocation takes one host physical lease plus a non-local
residency allowance, not two physical leases.

An earlier version of this appendix recommended preferring "a resident hot set
plus host-mapped cold, single-touch weights over copy-map-evict churn." That
recommendation was built and measured, and **it does not hold on this class of
hardware** (#864/#912). Two independent limits defeat it:

- **Correctness.** Aggregate distinct host-mapped bytes read per decode step
  above ~0.44–0.65 GB silently returned stale data on an RTX 4060 Laptop —
  generation collapsed 16 tokens to 3 with no error raised. A *single*
  host-mapped read was bit-identical at 1, 8, 16 and 32 cold weights, and a
  copy-instead A/B isolated the fault to the host-mapped read rather than the
  admission flow, which places the ceiling on aggregate aperture. The failure
  is invisible: nothing errors, the answer is simply wrong.
- **Throughput.** Even capped at a provably safe 256 MiB budget, the hybrid ran
  **0.73 tok/s against WDDM demand paging's 7.84** on the same model in the
  same session. The safe aperture (0.26 GB/step) is structurally smaller than
  the traffic it would have to displace (~0.6 GB/step), leaving the lever short
  by construction.

The claim that survives is narrower: **prefer a resident hot set, and let the
platform move the cold remainder** unless C7 reports a host-mapped read capacity
large enough to cover the per-step cold traffic. The mechanism itself is sound —
capture survives host-mapped pointers with bit-identical replay — so this stands
as a capacity verdict about one class of GPU.

**And the re-measurement has since happened, with the opposite result (#925).**
On an H200 under Linux (driver 580.105.08, CUDA 13, kernel 6.6) the aperture
ceiling is **absent**: generation stayed byte-identical to baseline with
`fallbacks=0` up to **6.795 GB** of distinct host-mapped weights bound and
re-read in place every decode step — 704 `cuMemHostRegister` binds, n=3, all
runs byte-identical, ~15× the WDDM ~0.44 GB onset and ~10× the top of its
corruption band. There the hybrid is worth **~8× (67 tok/s against ~8.5 median
for managed streaming)**, because Linux has no OS fallback and the competitor is
managed streaming or failure rather than a fast demand-paging path. The default
budget is now platform-conditional — 256 MiB on Windows, 2 GiB elsewhere,
bounded at >3× under the measured-safe figure since only one GPU class was
tested (#936).

This pair is the clearest case for putting the capacity in C7 rather than in a
constant: the same capability differs by more than an order of magnitude across
two platforms, and getting it wrong costs silently corrupted output in one
direction and ~8× of throughput in the other.

Managed no-spill CUDA VMM is a hard bound on *our own admission* — the ledger
refuses to hand out more than `managed_limit`, and a request past it fails
rather than silently degrading. It is **not** a guarantee that the granules we
did admit stay in device memory. #863 measured WDDM paging out our own VMM
granules under system-wide over-commit, so the correct statement is narrower:
solo and under `managed_limit`, no-spill holds physically (`nvidia-smi` tracks
our ledger 1:1); once the *system* is over-committed, VidMm may demote our
granules and our ledger cannot see it. Neither we nor WDDM can pin against
that on this platform. The design consequence is unchanged — never treat WDDM
spill as a capacity extension we can plan against — but the reason is that
spill is invisible and slow, not that it cannot happen to us. Under TCC,
`cuMemCreate` should fail rather than spill (#783).

CUDA managed memory is a separate capability with limited Windows support.

## G. Prefix caching, batching, and model switching

| Behavior | Requirement |
|---|---|
| Prefix caching | Cache an immutable, content-addressed complete state bundle. Key by model, tokenizer, adapter, positions, dtype/layout, and schema. Share blocks/granules once; COW at divergence. Hybrid KV-only hits recompute. |
| Continuous batching | Reserve each selected request's state delta and transient peak. Preserve completion-feasible admission and atomically commit all state/search progress. |
| Model switching | Demote/discard reconstructible weights, graph pools, and prefixes under model leases. Protect active requests and require residency-progress admission. |

Current evidence is uneven. VMM prefix sharing and native recurrent-prefix
restore have parity/accounting tests. ORT hybrid restore correctly recomputes.
Paged KV transaction logic exists, but device-backed `onnx-genai-kv` integration
with native CUDA is not implemented.

## H. Related work

| Reference | Precedent | Limitation relative to this design |
|---|---|---|
| [vLLM](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/block_pool.py) | Fixed-block paged KV, sharing, eviction. | Per-engine utilization; no shared machine authority. |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/kvcache.md) | Paged KV, reuse, host offload. | Geometry pools are statically split. |
| [llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/common/fit.cpp) | Whole-model/KV/compute fit across backends. | Startup snapshot, not runtime multi-model authority. |
| [Accelerate](https://huggingface.co/docs/accelerate/concept_guides/big_model_inference) | Static GPU/CPU/disk device map. | Weight-centric; no runtime state/activation leases. |
| [MLX](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html) | One allocator over Apple UMA. | Process-scoped; Apple-specific topology. |
| [vAttention](https://arxiv.org/abs/2405.04437) | Flat contiguous VA with physical pages on demand. | Research prototype; stock CUDA granularity remains material. |

## I. Evidence and risk details

| Risk | Confidence in mitigation | Repository evidence / remaining gate |
|---|---|---|
| Incomplete accounting | Medium | Governed ORT/CUDA weight/KV/workspace leases exist; activation/overhead remain incomplete, and #628 still substitutes token count for bytes when geometry is unknown. |
| Governance overhead | Medium | Header host path measured about 15 ns; VMM charges on granule growth. Multi-session/device-arena contention remains. |
| VMM granularity | Medium-high | 2 MiB CUDA granule and layout crossovers measured. Against the engine's actual packed head-major floor, token-major would reduce the floor by about 96x and is not implemented. The earlier 768x probe used a fixed full-context stride the engine does not instantiate. |
| Remap/commit safety | Medium | Same-VA replay, stable slots, prefix multi-map, and accounting pass tests. In-flight unmap and multi-model/multi-stream stress remain. |
| Dynamic weight lending | **Blocked** | Large stable-slot eviction/re-admission corrupts output. Static hot set only until isolated and token-parity gated. Eviction *order* is ruled out as the cause (#892) and bounded at ~10% of the gap (#901); the open lever is admission. |
| Host-mapped cold weight reads | **Negative, closed** | Built end-to-end and measured (#912). Single reads bit-identical; aggregate distinct reads above ~0.44–0.65 GB/step silently corrupt; safe-capped arm ran 0.73 tok/s against WDDM's 7.84. Capture with host-mapped pointers works and replays bit-identically, so the mechanism is sound and the *capacity* is not. Re-measure on parts with larger host apertures; do not assume the result transfers either way. |
| Managing vs. deferring to the platform | High | Direct A/B on a 14B model over budget: managed streaming 0.18 tok/s vs WDDM demand paging 5.53 (#864); shipping the default change measured ~100x end-to-end (#874). The structural cause is that each weight is read once per decode step, so copying it buys no reuse. |
| Batching as the amortization lever | Medium-high | `1/N` amortization measured with batch-invariant totals and a KV-content control at `past_len` 0/512/2048 (#884/#891); ceiling `N_max ~ 19 @ 2048 ctx`. Requires one fused forward with `M = N`; `N` sequential forwards amortize nothing. |
| Weight-byte accounting | High | Corrected 2.00x by summing referenced extents rather than the external-data blob (#853/#856); the error was found because measured traffic sat below its own theoretical floor. |
| Static KV reservation | High | A load-time `bytes_per_token x max_context` split was charged in full from the first token; making it elastic cut streamed bytes 1.68x with the max-context guarantee tested through the production reclaim path (#857/#866). |
| Pressure liveness | Medium | Single HostGovernor protocol has TLA/refinement and conformance. Multi-authority ordering, persistent non-owning intents, bounded-aging arbitration, and real-holder fault campaigns remain. |
| State completeness | Medium | Native recurrent prefix parity exists; ORT hybrid reuse recomputes until full state restore lands. |
| Coordinator quota leak | Medium-low | Authenticate spawned workers; heartbeat/epoch failure fences and initiates termination; quota remains charged until process exit; return is idempotent across restart. |
| External consumers | Low | Telemetry/safety reserve only; unrelated programs cannot be reclaimed. |
| Device loss/provider teardown | Low | No complete implementation today; needs terminal authority generation and provider pinning. |

## J. Current implementation gap

Today the design is only partially realized:

- native CPU/CUDA EP instances own allocator handles;
- `onnx-runtime-memory-api::BindingRegistry` now provides narrow manager-issued
  device/mechanism/context/authority bindings, lifetime pins, switch
  stability, allocation generations, and device-loss invalidation; existing
  EP paths adopt it incrementally rather than changing policy;
- the server shares device authorities, but host/disk ledgers remain per engine;
- no process-wide `ProcessMemoryManager` composes that registry with canonical
  authorities, holders, policy, and transactions yet;
- ORT governed allocator lifetime uses a bounded leak because the last session
  user is not observable;
- `onnx-genai-kv` device storage is not wired into native CUDA decode;
- `ProcessMemoryManager`, `ServingMemoryCoordinator`, and the cross-authority
  transaction coordinator are proposed.

## References

- [`MEMORY_ARCHITECTURE.md`](MEMORY_ARCHITECTURE.md)
- [`PRESSURE_PROTOCOL_IMPL.md`](PRESSURE_PROTOCOL_IMPL.md)
- [`WEIGHT_OFFLOAD.md`](WEIGHT_OFFLOAD.md)
- [`native-ort-kv-capacity.md`](native-ort-kv-capacity.md)
- [`KvAdmission.tla`](../../specs/tla/KvAdmission.tla)
- [`CoResidency.tla`](../../specs/tla/CoResidency.tla)
- [ORT memory consumption](https://onnxruntime.ai/docs/performance/tune-performance/memory.html)
- [ORT shared allocators and arena shrinkage](https://onnxruntime.ai/docs/get-started/with-c.html#features)
- [ORT CUDA EP options](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
- [ORT device tensors](https://onnxruntime.ai/docs/performance/device-tensor.html)
- [WDDM GPU virtual memory](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/gpu-virtual-memory-in-wddm-2-0)
- [DXGI video-memory budgets](https://learn.microsoft.com/en-us/windows/win32/api/dxgi1_4/ns-dxgi1_4-dxgi_query_video_memory_info)
- [CUDA Unified Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#unified-memory)
