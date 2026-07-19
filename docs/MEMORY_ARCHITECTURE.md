# Memory Architecture: Unified Design

> Consolidates memory management design from
> [WEIGHT_OFFLOAD.md](./WEIGHT_OFFLOAD.md),
> [DESIGN.md](./DESIGN.md) §26.11 & §43.2,
> [MOE_SUPPORT.md](./MOE_SUPPORT.md) §7,
> [MOE_EXPERT_PARALLELISM.md](./MOE_EXPERT_PARALLELISM.md) §8, and
> [DISTRIBUTED_RUNTIME.md](./DISTRIBUTED_RUNTIME.md) §3, §5, §12.

**Status:** Design — Consolidated
**Author:** Claw (with Justin)
**Date:** 2026-07-19

---

## Table of Contents

1. [Overview](#1-overview)
2. [Layer 1: EP Memory (Device-Local)](#2-layer-1-ep-memory-device-local)
3. [Layer 2: Weight Residency (Per-Session)](#3-layer-2-weight-residency-per-session)
4. [Layer 3: Resource Governor (Per-Device, Cross-Session)](#4-layer-3-resource-governor-per-device-cross-session)
5. [Layer 4: Memory Coordinator (Cross-Node, genai-server)](#5-layer-4-memory-coordinator-cross-node-genai-server)
6. [Communication Layer](#6-communication-layer)
7. [Heterogeneous Device Support](#7-heterogeneous-device-support)
8. [Decision Log](#8-decision-log)
9. [Phased Implementation](#9-phased-implementation)
10. [Open Questions](#10-open-questions)
11. [References](#11-references)

---

## 1. Overview

Memory management in onnx-genai is organized as a four-layer hierarchy. Each layer
has a distinct scope, a distinct owner, and a distinct reason to exist:

```text
┌─────────────────────────────────────────────────────────────────────────┐
│  Layer 4: MemoryCoordinator   (cross-node, genai-server)               │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  Layer 3: ResourceGovernor (per-DEVICE, cross-session)            │  │
│  │  ┌─────────────────────────────────────────────────────────────┐  │  │
│  │  │  Layer 2: WeightResidencyManager (per-session)              │  │  │
│  │  │  ┌───────────────────────────────────────────────────────┐  │  │  │
│  │  │  │  Layer 1: EP Memory (device-local allocate/free)      │  │  │  │
│  │  │  └───────────────────────────────────────────────────────┘  │  │  │
│  │  └─────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

**Why four layers?**

| Layer | Scope | Question it answers |
|---|---|---|
| 1. EP Memory | One device, one allocation | "Where do these bytes live on this device?" |
| 2. Weight Residency | One session, one model | "Which weight regions are cold/warm/hot?" |
| 3. Resource Governor | One device, all sessions | "How much memory can each session use?" |
| 4. Memory Coordinator | All devices, all nodes | "How should memory be shared across machines?" |

Layers compose bottom-up: the coordinator calls into governors, governors constrain
residency managers, and residency managers allocate through EPs. No layer bypasses
the one below it.

---

## 2. Layer 1: EP Memory (Device-Local)

The `ExecutionProvider` trait exposes raw device memory operations. This is the
lowest layer — purely local, no cross-session awareness, no policy.

**Reference:** `crates/onnx-runtime-ep-api/src/provider.rs`

```rust
pub trait ExecutionProvider: Send + Sync {
    /// Allocate `size` bytes of device memory.
    fn allocate(&self, size: usize, alignment: usize) -> Result<DeviceBuffer>;

    /// Free a previously allocated device buffer.
    fn deallocate(&self, buffer: DeviceBuffer) -> Result<()>;

    /// Copy bytes between host and device, or device-to-device.
    fn copy(&self, src: &DeviceBuffer, dst: &mut DeviceBuffer, size: usize) -> Result<()>;

    // ... other EP methods (execute, supports_op, etc.)
}
```

Every higher layer ultimately calls through these primitives. The EP knows nothing
about weight tiers, budgets, or other sessions.

---

## 3. Layer 2: Weight Residency (Per-Session)

Treats immutable model weights as a three-tier hierarchy within a single session.
This is the design from [WEIGHT_OFFLOAD.md](./WEIGHT_OFFLOAD.md), consolidated here.

### 3.1 Components

```text
ONNX loader / WeightStore
  owns read-only mmaps and validated WeightRef ranges
                 |
                 v
WeightRegionCatalog
  classifies shared tensors and expert subranges; records format/layout/alignment
                 |
                 v
WeightResidencyManager  <---- Resource Governor sub-budgets (Layer 3)
  cold mmap | warm host pages | hot device pages | LRU/heat | in-flight state
                 |
         +-------+--------+
         |                |
         v                v
ExpertStore facade    static layer placement
fused MoE kernels     dense/attention/embedding/lm-head bindings
```

### 3.2 Interfaces

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
    /// Acquire a lease on a weight region. The lease pins the region in its
    /// current tier, preventing eviction until dropped.
    fn lease(&self, request: WeightRequest) -> Result<WeightLease>;

    /// Speculatively begin loading a weight region toward a warmer tier.
    fn prefetch(&self, request: WeightRequest);

    /// Report observed expert routing for heat-based admission.
    fn observe_routes(&self, layer: usize, experts: &[u32]);

    /// Current residency state for monitoring/debugging.
    fn usage(&self) -> WeightResidencySnapshot;
}

trait ExpertStore {
    /// Ensure the given experts are resident on `target` device.
    /// Returns a lease that pins them until dropped.
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

### 3.3 Tier Semantics

#### Cold: Read-Only mmap Backing

- Canonical bytes are ONNX external data and remain immutable.
- A cold hit returns a checked subrange of the existing mmap.
- CPU direct-compressed kernels may consume that range without a host copy.
- Clean mapped pages can be discarded after use; strict budget reporting must
  distinguish owned host-cache bytes from OS page-cache/RSS.

Inline initializers are acceptable for small shared tensors, but offloadable expert
pools must use external data.

#### Warm: Bounded Host RAM

Warm entries are optional derived copies of canonical packed pages:

- pageable aligned pages for CPU reuse;
- pinned pages for repeated H2D transfer;
- optional CPU-prepacked or dequantized panels only when their expanded byte cost is
  charged to the host budget and measured reuse justifies it.

The warm cache uses byte-based LFRU admission with hysteresis, not entry count. A miss
always falls back to mmap, so a zero-byte host cache remains functional.

#### Hot: Bounded Device VRAM

A device entry is an EP-owned allocation containing either canonical compressed bytes
or an explicitly versioned device-prepacked representation. It is keyed by
`(region, representation, device)` and charged at actual allocated bytes. Eviction is
legal only when no lease or transfer owns the entry. Failed speculative prefetch must
not displace a leased or demonstrably hotter entry.

On a fully resident plan, entries are pinned for the session and the manager collapses
to stable pointer lookup.

### 3.4 Expert Paging and Batching

For one admitted token batch:

1. Run the model-exact router and compute exact top-k IDs and aggregation weights.
2. Union selected expert IDs across all token rows.
3. Group rows by expert and compute token counts.
4. Ask `ExpertStore` for a residency plan under the current tier budgets.
5. Execute resident experts together; process the remainder in bounded waves/tiles.
6. Scatter and combine with the original aggregation weights.
7. Release CPU leases immediately and device leases after completion fences signal.
8. Record routes, bytes, stalls, and reuse for future admission/prefetch.

**Expert is the policy unit; page/tile is the capacity unit.** A whole expert is
convenient for heat and LRU decisions but may itself exceed free RAM/VRAM. Store each
expert contiguously, then divide its FC1/FC2/FC3, scale, zero-point, and bias ranges
into page-aligned transfer tiles.

- **Admission:** choose experts by heat/priority.
- **Transfer:** move bounded pages/tiles.
- **Compute:** consume direct compressed blocks or double-buffered panels.
- **Atomicity:** a logical expert lease groups all companion ranges required by the
  current kernel wave; it does not imply the whole expert is copied at once.

### 3.5 Residency Policy

- Shared attention, router, normalization, embeddings, and other dense weights have
  higher base priority than routed experts because they are touched predictably.
- Expert admission combines frequency, recency, bytes, measured load cost, and tokens
  served while resident. Use hysteresis to avoid ping-pong.
- A page used by an in-flight kernel or transfer is non-evictable.
- Derived dequantized/prepacked entries are disposable and never the sole copy.

### 3.6 Cache Policy and Prefetch

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

---

## 4. Layer 3: Resource Governor (Per-Device, Cross-Session)

The Resource Governor is the engine-level byte-budget authority.
**CRITICAL: there is exactly one governor per DEVICE, not per session.** It is
shared across all sessions on that device. The full design lives in
[DESIGN.md §26.11](./DESIGN.md) and remains canonical there; this section
summarizes the key interfaces and how the governor integrates with weight
residency and expert stores.

### 4.1 User-Facing Limit Model

Each limit is expressible three ways — absolute bytes, fraction, or auto:

```rust
#[derive(Debug, Clone, Copy)]
pub enum ResourceLimit {
    Bytes(u64),
    Fraction(f32),   // of detected tier capacity
    Auto,            // sane default (90% VRAM, 25% host RAM)
}

#[derive(Debug, Clone)]
pub struct ResourceLimits {
    pub vram_limit: ResourceLimit,
    pub host_ram_limit: ResourceLimit,
    pub disk_spill_limit: Option<ResourceLimit>,
}
```

### 4.2 Live Reconfigurability

```rust
impl ResourceGovernor {
    pub fn reconfigure(&self, limits: ResourceLimits) -> Result<ReconfigureOutcome, ResourceError>;
    pub fn set_vram_limit(&self, limit: ResourceLimit) -> Result<ReconfigureOutcome, ResourceError>;
    pub fn set_host_ram_limit(&self, limit: ResourceLimit) -> Result<ReconfigureOutcome, ResourceError>;
    pub fn set_disk_spill_limit(&self, limit: Option<ResourceLimit>) -> Result<ReconfigureOutcome, ResourceError>;
    pub fn snapshot(&self) -> GovernorSnapshot;
}
```

Limits can change mid-session without restart. The governor holds limits behind
`ArcSwap<ResolvedLimits>` for lock-free reads on the hot admission path;
`reconfigure` serializes writers with a mutex.

### 4.3 Cross-Session Invariant

```rust
// Invariant checked on every reconfigure:
//   sum(session.max_pages or actual) ≤ budget.total_pages
//   interactive_reserve = round(reserve_fraction × budget.total_pages)
//   every per-session cap ≤ budget.total_pages − interactive_reserve
```

A single runaway session cannot blow the global VRAM budget — all allocations go
through the same `can_allocate` gate, which the governor now bounds in bytes.

### 4.4 Tiered Eviction on Lowering

When a limit is lowered below current usage, the governor drives existing eviction
tiers in order:

1. Drop **background** sessions' KV (cheap to re-prefill).
2. Offload **paused standard** sessions' KV to the warm tier.
3. Preempt **running standard** sessions (recompute from last checkpoint on resume).
4. **Interactive** sessions and `interactive_reserve` are touched last.

The call blocks until under ceiling or tiers are exhausted. If the target cannot
be met, the governor **rejects atomically**, restores the previous ceiling, and
returns `ResourceError::CannotSatisfyLoweredLimit`.

### 4.5 VramBreakdown

The governor decomposes VRAM usage into trackable components:

```rust
pub struct VramBreakdown {
    pub model_weights_bytes: u64,      // dense weights
    pub hot_expert_cache_bytes: u64,   // hot expert cache (from ExpertStore)
    pub kv_cache_bytes: u64,           // KV cache pages
    pub activations_bytes: u64,        // peak activation working set
    pub ort_overhead_bytes: u64,       // arena / session / EP overhead
    pub total_bytes: u64,
}
```

**Constraint:** `dense_weights + hot_expert_cache + kv_cache + activations + overhead ≤ ceiling`

### 4.6 Sub-Budget Coordination: KV vs Expert LRUs

Independent KV and expert LRUs must not race for the last bytes. The governor
assigns coordinated sub-budgets and can rebalance them with hysteresis:

```text
VRAM ceiling = resident shared weights
             + hot expert/device cache
             + KV cache
             + activations and routing scratch
             + EP/runtime overhead
```

Both the `WeightResidencyManager` and KV cache manager receive sub-budgets from
the governor and return usage. On lowering a live limit: cancel speculative
reservations, evict unleased weight pages, demote KV, reduce batch/scratch, and
return an actionable minimum-working-set error if still impossible.

> See [DESIGN.md §26.11](./DESIGN.md) for the full governor design including
> config surfaces (YAML + Rust API + Python), error experience
> (`ResourceError` with what/why/how), and implementation status.

> See [DESIGN.md §43.2](./DESIGN.md) for the declaration that expert weights
> are "not KV cache" and the rationale for separate APIs with shared concepts.

### 4.7 Config Surface

**YAML** (extends the `memory:` block in `server_config.yaml`):

```yaml
serving:
  memory:
    limits:
      vram_limit: "8GiB"            # absolute; or "0.9" (fraction); or "auto"
      host_ram_limit: "16GiB"
      disk_spill_limit: null         # null = disabled (default)
      allow_runtime_override: true   # permit live reconfigure via API
    interactive_reserve_pct: 20
    eviction_policy: priority_then_lru
    offload_to_cpu: true
```

**Rust:**

```rust
let engine = GenAiEngine::load(model, EngineConfig { limits, .. })?;
engine.governor().set_vram_limit(ResourceLimit::Bytes(6 << 30))?;
let snap = engine.governor().snapshot();
```

**Python:**

```python
engine.set_vram_limit("6GiB")
snap = engine.resource_snapshot()  # dict: per-tier used / limit / headroom
```

### 4.8 Error Experience

```rust
pub enum ResourceError {
    VramOverBudget {
        requested_bytes: u64,
        limit_bytes: u64,
        available_bytes: u64,
        breakdown: VramBreakdown,
        tier: Tier,
        suggestions: Vec<Remedy>,
    },
    CannotSatisfyLoweredLimit {
        requested_limit_bytes: u64,
        floor_bytes: u64,
        breakdown: VramBreakdown,
        reclaimable_bytes: u64,
        suggestions: Vec<Remedy>,
    },
    SessionLimitExceedsGlobal {
        session: SessionId,
        requested_pages: usize,
        global_pages: usize,
    },
}
```

---

## 5. Layer 4: Memory Coordinator (Cross-Node, genai-server)

### 5.1 When Is This Needed?

- **Single-machine, single-session:** Not needed. The governor (Layer 3) is sufficient.
- **Single-machine, multi-session:** The governor IS the coordinator. One governor per
  device already enforces `sum(session.usage) ≤ ceiling`. No additional coordination
  layer is required.
- **Multi-node distributed deployment:** The `MemoryCoordinator` is needed to coordinate
  across governors on different machines.

**Key clarification:** For single-machine multi-session, the per-device `ResourceGovernor`
already handles cross-session budgeting (§4.3). The `MemoryCoordinator` adds value only
when cross-session *optimizations* are desired (shared weight dedup, KV prefix sharing,
expert migration) or when multiple physical nodes must coordinate.

### 5.2 Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                     genai-server                                 │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  MemoryCoordinator (global, per-machine)                   │  │
│  │                                                            │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │  │
│  │  │ Weight Dedup  │  │ KV Pool      │  │ Budget Arbiter  │  │  │
│  │  │ (shared mmap) │  │ (prefix      │  │ (rebalance      │  │  │
│  │  │               │  │  sharing)    │  │  sub-budgets)   │  │  │
│  │  └──────────────┘  └──────────────┘  └─────────────────┘  │  │
│  └───────┬───────────────────┬───────────────────┬────────────┘  │
│          │                   │                   │               │
│    ┌─────▼─────┐       ┌─────▼─────┐       ┌─────▼─────┐       │
│    │ Session 0 │       │ Session 1 │       │ Session N │       │
│    │           │       │           │       │           │       │
│    │ Governor  │       │ Governor  │       │ Governor  │       │
│    │ Residency │       │ Residency │       │ Residency │       │
│    │ KV cache  │       │ KV cache  │       │ KV cache  │       │
│    └───────────┘       └───────────┘       └───────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 MemoryCoordinator Interface

```rust
/// Global memory coordinator across sessions on one machine.
///
/// Sits above per-session ResourceGovernors, adjusting their budgets
/// and providing cross-session optimizations.
trait MemoryCoordinator: Send + Sync {
    // ── Shared Weight Deduplication ──

    /// Register a weight region for deduplication. Returns a handle
    /// that multiple sessions can use without each allocating a copy.
    /// Uses CUDA IPC / mmap for zero-copy sharing.
    fn register_shared_weight(
        &self,
        region: &WeightRegion,
        device: DeviceId,
    ) -> Result<SharedWeightHandle>;

    /// Acquire a read-only view of a shared weight. Ref-counted;
    /// the weight stays resident as long as any session holds a view.
    fn acquire_shared_view(
        &self,
        handle: &SharedWeightHandle,
        session: SessionId,
    ) -> Result<WeightView>;

    // ── Cross-Session KV Cache ──

    fn request_kv_pages(
        &self,
        session: SessionId,
        num_pages: usize,
        priority: PagePriority,
    ) -> Result<Vec<PageHandle>>;

    fn release_kv_pages(&self, pages: Vec<PageHandle>);

    fn lookup_prefix(
        &self,
        token_hash: u64,
        num_tokens: usize,
    ) -> Option<PrefixCacheHit>;

    // ── Expert Migration ──

    fn migrate_expert(
        &self,
        expert: ExpertId,
        from: SessionId,
        to: SessionId,
    ) -> Result<()>;

    fn report_expert_heat(
        &self,
        session: SessionId,
        layer: usize,
        activations: &[(ExpertId, u32)],
    );

    // ── Budget Arbitration (drives Layer 3 governors) ──

    fn memory_pressure(&self) -> MemoryPressure;

    /// Rebalance sub-budgets across sessions.
    /// Pushes adjustments down to each session's ResourceGovernor
    /// via `governor.reconfigure()`.
    fn rebalance(&self) -> Vec<BudgetAdjustment>;

    fn set_session_limit(
        &self,
        session: SessionId,
        limit: ResourceLimit,
    ) -> Result<ReconfigureOutcome>;
}

struct BudgetAdjustment {
    session: SessionId,
    new_kv_budget_bytes: usize,
    new_expert_cache_bytes: usize,
    reason: AdjustmentReason,
}

enum AdjustmentReason {
    KvPressure { requesting_session: SessionId },
    ExpertHeatShift,
    GlobalPressure,
}
```

### 5.4 How Coordinator Calls Down Into Governors

```text
MemoryCoordinator.rebalance()
  │
  ├── reads: governor[0].snapshot() → {used: 120GB, limit: 141GB, headroom: 21GB}
  ├── reads: governor[1].snapshot() → {used: 139GB, limit: 141GB, headroom: 2GB}
  │   └── GPU 1 under pressure!
  │
  ├── decides: GPU 1 needs 15GB for KV. GPU 0 has 21GB headroom.
  │   Migrate cold expert 742 (3GB) from GPU 1 → GPU 0.
  │   Lower GPU 1's expert sub-budget by 3GB, raise KV sub-budget.
  │
  ├── calls: governor[1].reconfigure({vram_kv: +3GB, vram_expert: -3GB})
  │   └── Governor triggers tiered eviction on expert cache
  │
  └── calls: governor[0].reconfigure({vram_expert: +3GB})
      └── Governor admits the migrated expert
```

The coordinator never sets a per-session limit that would violate the global ceiling.
If it tries, the governor rejects with `ResourceError::CannotSatisfyLoweredLimit`
and the coordinator rolls back.

### 5.5 Three Progressive Strategies

#### Strategy 1: Static Isolation

Each session gets a fixed budget. No cross-session coordination. The per-session
`ResourceGovernor` operates unchanged. Identical to running independent processes.

#### Strategy 2: Shared Weights + Shared KV Pool

Deduplicate shared weights (attention/router/embed stored ONCE via CUDA IPC);
unify KV cache pool. This is where `register_shared_weight()` and
`request_kv_pages()` become active.

```text
8×H200, 1128 GB total:
  Shared weights (attention/router/embed): 50 GB (stored ONCE via CUDA IPC)
  Expert weights (per-session shard): 700 GB
  KV cache (global pool): 350 GB  ← was 8×43=344 GB, now unified
  Scratch: 28 GB
  Savings: 7 × 50 GB = 350 GB freed from weight duplication
```

#### Strategy 3: Dynamic Expert Migration + Replication

The coordinator monitors expert heat and actively rebalances — replicating hot
experts across GPUs and evicting cold ones. Extends the `observe_routes()`
mechanism from the per-session residency manager.

### 5.6 Cross-Node Memory Coordination

For multi-node (e.g., Mac Studio cluster), the coordinator splits into:

- **LocalCoordinator** per machine — handles CUDA IPC / mmap sharing.
- **GlobalCoordinator** in genai-server — handles cross-node expert migration
  (via Communicator), cross-node prefix cache lookup, and global budget
  arbitration.

```text
┌─ Node 0 ─────────────────┐    ┌─ Node 1 ─────────────────┐
│ LocalCoordinator          │    │ LocalCoordinator          │
│ ├── Session 0 (GPU 0)     │    │ ├── Session 2 (MLX)       │
│ └── Session 1 (GPU 1)     │    │ └── Session 3 (MLX)       │
└───────────┬───────────────┘    └───────────┬───────────────┘
            │                                │
            └───────── GlobalCoordinator ─────┘
                       (in genai-server)
```

Cross-node expert migration transfers weights via the Communicator (§6). Within a
node, shared weights use zero-copy IPC. The `GlobalCoordinator` delegates intra-node
sharing to the `LocalCoordinator`.

---

## 6. Communication Layer

The `Communicator` trait is the runtime-level communication abstraction for
distributed inference. It lives alongside EPs in the runtime — EPs produce
tensors; the Communicator moves them between devices. Full design in
[DISTRIBUTED_RUNTIME.md §3](./DISTRIBUTED_RUNTIME.md).

### 6.1 Core Trait

```rust
#[async_trait]
pub trait Communicator: Send + Sync {
    fn rank(&self) -> Rank;
    fn world_size(&self) -> usize;
    fn backend_name(&self) -> &str;

    // ── Collectives ──
    async fn all_reduce(&self, tensor: &mut DeviceBuffer, len: usize, dtype: DType, op: ReduceOp) -> Result<()>;
    async fn all_to_all(&self, send_bufs: &[&DeviceBuffer], recv_bufs: &mut [&mut DeviceBuffer], chunk_sizes: &[usize], dtype: DType) -> Result<()>;
    async fn all_gather(&self, send_buf: &DeviceBuffer, recv_buf: &mut DeviceBuffer, count: usize, dtype: DType) -> Result<()>;
    async fn broadcast(&self, buffer: &mut DeviceBuffer, len: usize, dtype: DType, root: Rank) -> Result<()>;
    async fn reduce_scatter(&self, send_buf: &DeviceBuffer, recv_buf: &mut DeviceBuffer, count: usize, dtype: DType, op: ReduceOp) -> Result<()>;

    // ── Point-to-point ──
    async fn send(&self, buffer: &DeviceBuffer, len: usize, dtype: DType, dest: Rank) -> Result<()>;
    async fn recv(&self, buffer: &mut DeviceBuffer, len: usize, dtype: DType, source: Rank) -> Result<()>;

    // ── Synchronization ──
    async fn barrier(&self) -> Result<()>;
}

pub struct Rank(pub u32);

pub enum ReduceOp { Sum, Product, Min, Max }
```

### 6.2 Five Backends

```text
┌──────────────────────┬──────────────┬───────────────┬──────────────┐
│ NcclCommunicator     │ GlooComm     │ ThunderboltCm │ InProcessCm  │
│                      │              │               │              │
│ Multi-GPU, NVLink    │ CPU + TCP    │ Mac Studio    │ Testing /    │
│ PCIe, NVSwitch       │ ethernet     │ TB5 RDMA      │ simulation   │
│ 900 GB/s (NVLink)    │ 1-25 GB/s    │ ~12 GB/s      │ memcpy       │
│ <1 μs latency        │ ~100 μs      │ ~5 μs         │ ~0 μs        │
├──────────────────────┴──────────────┴───────────────┴──────────────┤
│ RdmaCommunicator     │                                             │
│ InfiniBand / RoCE    │  Data center cross-node, 200-400 Gbps       │
└──────────────────────┴─────────────────────────────────────────────┘
```

- **NcclCommunicator** — NVIDIA multi-GPU. Operates directly on CUDA device buffers.
- **GlooCommunicator** — CPU tensors over TCP/IP. Fallback and control-plane coordination.
- **ThunderboltCommunicator** — Mac Studio clusters via TB5 RDMA (~12 GB/s). Operates on
  host-accessible unified memory (MLX).
- **RdmaCommunicator** — InfiniBand/RoCE. Supports GPUDirect RDMA.
- **InProcessCommunicator** — All ranks in-process via memcpy. Testing and simulation.

### 6.3 Communicator Supersedes DispatchTransport

> **DEPRECATED:** The `DispatchTransport` trait from
> [MOE_EXPERT_PARALLELISM.md §8](./MOE_EXPERT_PARALLELISM.md) is superseded by
> `Communicator`. `DispatchTransport` was MoE-specific (send/recv/all_reduce/all_to_all
> scoped to expert dispatch). `Communicator` generalizes this to support tensor
> parallelism, pipeline parallelism, and expert parallelism through a single interface.
>
> **Use `Communicator` for all new work.** `DispatchTransport` remains in
> MOE_EXPERT_PARALLELISM.md for historical reference only.

The key differences:

| Aspect | DispatchTransport (deprecated) | Communicator |
|---|---|---|
| Scope | MoE expert dispatch only | All distributed patterns |
| Buffer type | `Tensor` (opaque) | `DeviceBuffer` with explicit dtype/len |
| Sub-groups | None | `CommGroup` for hybrid TP+EP strategies |
| Device awareness | Implicit | Explicit `TransportCapability` for staging |
| Backends | 3 (CUDA IPC, Host, Network) | 5 (NCCL, Gloo, TB5, RDMA, InProcess) |

### 6.4 Buffer Location Awareness

```rust
pub struct TransportCapability {
    pub send_from: Vec<DeviceType>,
    pub recv_into: Vec<DeviceType>,
    pub staging_device: DeviceId,  // fallback when device type unsupported
}
```

---

## 7. Heterogeneous Device Support

Because communication lives outside EP, different EP types coexist naturally.
Full design in [DISTRIBUTED_RUNTIME.md §5](./DISTRIBUTED_RUNTIME.md).

### 7.1 Format Negotiation at Boundaries

```rust
pub struct TensorFormat {
    pub dtype: DType,
    pub layout: TensorLayout,
    pub quantization: Option<QuantFormat>,
}

pub struct FormatConverter {
    pub source: TensorFormat,
    pub target: TensorFormat,
    pub convert_on: DeviceId,
}
```

The runtime inserts format conversion nodes at EP boundaries automatically.

### 7.2 Mixing Scenarios

| Scenario | Devices | Communicator | Use Case |
|---|---|---|---|
| Multi-GPU single node | 8× H200, CUDA EP | NCCL | TP + EP for large models |
| Mac Studio cluster | 4× M3 Ultra, MLX EP | Thunderbolt | EP across Macs |
| Hybrid GPU+Mac | H200 + Mac Studio | Gloo (TCP) | Overflow to Mac for cold experts |
| NPU + GPU | NPU EP + CUDA EP | InProcess | NPU handles attention, GPU handles FFN |
| Multi-vendor GPU | ROCm EP + CUDA EP | Gloo/RDMA | Rare but architecturally possible |
| Dev/test | Multiple CPU EPs | InProcess | Verify distributed logic locally |

---

## 8. Decision Log

Key architectural decisions and their rationale:

### D1: Governor is per-device, not per-session

**Decision:** One `ResourceGovernor` per physical device, shared across all sessions.

**Rationale:** A per-session governor cannot enforce `sum(session.usage) ≤ device_capacity`.
The device-level governor is the single source of truth for byte budgets. Per-session
sub-limits nest under the global ceiling.

### D2: Communication outside EP, not inside

**Decision:** The `Communicator` trait lives alongside EPs in the runtime, not inside
the EP trait.

**Rationale:** EPs produce tensors; the Communicator moves them. This separation
enables heterogeneous deployment (CUDA EP + MLX EP in the same distributed graph)
and keeps EP implementations focused on compute.

### D3: Expert weights are not KV cache

**Decision:** Expert weights get a separate `ExpertStore` / `WeightResidencyManager`
API, not storage in `onnx-genai-kv`.

**Rationale:** Expert weights are immutable model data with different access patterns
(heat-based LRU, expert-major layout, read-only). KV cache is mutable, sequence-keyed,
and copy-on-write. They share *concepts* (tiering, leases, LRU, page tables) but not
identity, keys, or mutability semantics.

### D4: DispatchTransport → Communicator (superseded)

**Decision:** The MoE-specific `DispatchTransport` trait is deprecated in favor of
the general `Communicator` trait.

**Rationale:** `DispatchTransport` was scoped to MoE expert dispatch. Tensor
parallelism and pipeline parallelism need the same primitives. One trait covers
all distributed patterns with sub-groups for hybrid strategies.

### D5: MemoryCoordinator only for cross-node; single-machine uses governor

**Decision:** For single-machine multi-session, the per-device `ResourceGovernor` is
the coordinator. The `MemoryCoordinator` adds value only for cross-session
optimizations (shared weight dedup, KV prefix sharing) or multi-node coordination.

**Rationale:** Avoid adding a coordination layer where the governor already enforces
the invariant. The coordinator is an optimization layer, not a correctness requirement
for single-machine deployments.

### D6: ONNX multi-device annotations are hints, not execution constraints

**Decision:** The ONNX IR v11+ multi-device spec (`DeviceConfigurationProto`,
`ShardingSpecProto`, `NodeDeviceConfigurationProto`) is preserved in the IR as
optional annotations. The runtime reads them as **placement hints** for the graph
partitioner, but the `ParallelStrategy` makes the actual placement decision.

**What the ONNX spec provides:**
- `DeviceConfigurationProto` — model-level declaration of available device groups
  and their sizes.
- `NodeDeviceConfigurationProto` — per-node annotation of which device config it
  belongs to.
- `ShardingSpecProto` — per-tensor description of how axes are sharded across
  devices (shard vs replicate per dimension).

**How it interacts with our layers:**

```text
ONNX model (with optional sharding annotations)
    │
    ▼
Loader: parse DeviceConfigurationProto → NodeDeviceHints
    │
    ▼
IR: Node.device_hints (optional, informational)
    │
    ▼
ParallelStrategy: reads hints as ILP seed placement
    │  Hint says "TP on dim=0, 8 devices"
    │  → generate TensorParallel strategy, skip analysis
    │  No hint → fall back to automatic graph analysis
    │
    ▼
Communicator: executes communication (hint-agnostic)
    │
    ▼
EP: execute() (unaware of sharding)
```

**What we store in IR:**

```rust
struct NodeDeviceHints {
    /// Which device configuration this node prefers.
    pub config_name: Option<String>,
    /// Sharding specs for inputs/outputs.
    pub input_sharding: Vec<Option<ShardingSpec>>,
    pub output_sharding: Vec<Option<ShardingSpec>>,
}

struct ShardingSpec {
    /// Device IDs across which this tensor is sharded/replicated.
    pub devices: Vec<String>,
    /// Per-axis sharding description.
    pub sharded_dims: Vec<ShardedDim>,
}
```

**Rationale:**
- ONNX annotations are declarative ("SHOULD be sharded this way"), not imperative.
  They don't specify communication — that's the `Communicator`'s job.
- If Mobius or other exporters annotate models with sharding specs, the partitioner
  can skip expensive graph analysis and use the hints directly.
- The `onnx-rs` crate already validates these annotations (`MultiDeviceConfigurationRule`)
  but the runtime IR (`onnx-runtime-ir`) currently drops them after parsing. The
  loader should preserve them into `NodeDeviceHints` when present.
- Without annotations, the runtime falls back to automatic placement — no regression.

**Current status:** `onnx-rs` validates; IR/loader do not yet propagate. Low priority
until real models with sharding annotations exist.

---

## 9. Phased Implementation

Unified across all design documents:

### Phase 1: Single-Session Weight Residency

*Maps to WEIGHT_OFFLOAD.md Phases 1-2.*

- `WeightRegionCatalog` classifies model regions (shared vs expert).
- `WeightResidencyManager` with cold/warm/hot tiers.
- `ExpertStore` facade for fused MoE kernels.
- Heat-based LRU admission for experts.
- Lease/pin lifecycle with completion fences.
- Governor sub-budgets (KV vs expert) with hysteresis.

### Phase 2: Governor Wiring with Real EP/Model Usage

*Maps to DESIGN.md §26.11.*

- Connect real EP/model weight usage, activation/scratch high-water marks, and
  ORT/EP allocations to the governor.
- `hot_expert_bytes` component in `VramBreakdown`.
- Coordinated KV + expert sub-budget rebalancing.
- Lowering-triggered live eviction (tiered: background → paused → running → interactive).
- Auto mode with real capacity detection from EP device queries.

### Phase 3: Multi-GPU Single-Node

- NCCL `Communicator` for multi-GPU collective ops.
- Shared weights via CUDA IPC (zero-copy across sessions).
- `MemoryCoordinator` Strategy 2 (shared weights + shared KV pool).
- Expert migration between GPUs based on heat.
- InProcess `Communicator` for testing.

### Phase 4: Cross-Node

- Thunderbolt 5 `Communicator` for Mac Studio cluster.
- RDMA `Communicator` for data center.
- `GlobalCoordinator` above per-node `LocalCoordinator`.
- Cross-node expert migration via Communicator.
- Cross-node prefix cache lookup.

---

## 10. Open Questions

Consolidated from all source documents:

### From weight residency / governor (WEIGHT_OFFLOAD.md, DESIGN.md)

1. **Auto mode completeness.** Auto mode must not be considered complete until real
   free/total RAM, filesystem, and device capacity are reported by the EP.

2. **Budget reporting fidelity.** Clean mapped pages (cold tier) are OS page cache,
   not owned bytes. How to distinguish in budget reporting?

### From distributed coordination (DISTRIBUTED_RUNTIME.md)

3. **Rendezvous mechanism.** How do distributed ranks discover each other?
   Options: env vars (`MASTER_ADDR`/`MASTER_PORT`), shared file, TCP rendezvous
   server, mDNS/Bonjour.

4. **Fault tolerance.** What happens when a rank crashes mid-collective? NCCL
   aborts all ranks. Is restart-from-scratch acceptable for inference?

5. **Dynamic rank membership.** Can ranks join/leave a live session?

6. **Communicator selection.** When multiple backends are available, auto-select
   based on topology or user-configured?

7. **Quantized communication.** Send FP8/INT8 and up-cast at receiver to halve
   bandwidth?

8. **CUDA IPC ownership semantics.** When session 0 allocates shared weights and
   sessions 1-7 map via IPC, who owns the lifecycle? Options: dedicated weight
   server process, shared mmap-backed allocations, accept coupling.

9. **KV cache sharing granularity.** Different sessions may quantize KV differently
   (FP16 vs FP8). Enforce uniform format or support conversion at share boundaries?

10. **MemoryCoordinator placement.** In genai-server process or separate daemon?

### From MoE / expert parallelism

11. **Expert-aware scheduling across sessions.** When multiple sessions share a device,
    should the governor prefer expert affinity (co-locate sessions that use
    complementary expert sets)?

12. **Prefetch speculation budget.** How many speculative prefetch bytes before the
    cost of wrong predictions exceeds the benefit?

---

## 11. References

- [DESIGN.md §26.11](./DESIGN.md) — Resource Governor: canonical design (stays in place)
- [DESIGN.md §43.2](./DESIGN.md) — MoE Expert Weights: "not KV cache" declaration
- [WEIGHT_OFFLOAD.md](./WEIGHT_OFFLOAD.md) — Three-tier weight residency (redirects here for §4)
- [MOE_SUPPORT.md](./MOE_SUPPORT.md) — First-class MoE support (redirects here for §7)
- [MOE_EXPERT_PARALLELISM.md](./MOE_EXPERT_PARALLELISM.md) — Session-per-GPU MoE architecture (DispatchTransport deprecated)
- [DISTRIBUTED_RUNTIME.md](./DISTRIBUTED_RUNTIME.md) — Communicator abstraction & multi-device inference
- [SCHEDULING.md](./SCHEDULING.md) — Adaptive scheduling, EP negotiation protocol
- `crates/onnx-runtime-ep-api/src/provider.rs` — ExecutionProvider trait
- `crates/onnx-genai-scheduler/src/governor.rs` — ResourceGovernor implementation
