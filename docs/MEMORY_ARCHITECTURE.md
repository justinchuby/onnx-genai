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
4. [Layer 3a: DeviceGovernor (Per Compute Unit — Exclusive Memory)](#4-layer-3a-devicegovernor-per-compute-unit--exclusive-memory)
5. [Layer 3b: HostGovernor (Per Machine — Shared Memory)](#5-layer-3b-hostgovernor-per-machine--shared-memory)
6. [Layer 4: ClusterCoordinator (Cross-Node, genai-server)](#6-layer-4-clustercoordinator-cross-node-genai-server)
7. [Communication Layer](#7-communication-layer)
8. [Heterogeneous Device Support](#8-heterogeneous-device-support)
9. [Hardware Topology Variants](#9-hardware-topology-variants)
10. [Decision Log](#10-decision-log)
11. [Phased Implementation](#11-phased-implementation)
12. [Open Questions](#12-open-questions)
13. [References](#13-references)

---

## 1. Overview

Memory management in onnx-genai is organized as a five-layer hierarchy. Each layer
has a distinct scope, a distinct owner, and a distinct reason to exist:

```text
┌──────────────────────────────────────────────────────────────────────────┐
│  Layer 4: ClusterCoordinator  (cross-node, genai-server)                │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  Layer 3b: HostGovernor  (per MACHINE — shared host RAM + disk)    │  │
│  │  ┌──────────────────────────────────────────────────────────────┐  │  │
│  │  │  Layer 3a: DeviceGovernor  (per DEVICE — exclusive VRAM)     │  │  │
│  │  │  ┌────────────────────────────────────────────────────────┐  │  │  │
│  │  │  │  Layer 2: WeightResidencyManager (per-session)         │  │  │  │
│  │  │  │  ┌──────────────────────────────────────────────────┐  │  │  │  │
│  │  │  │  │  Layer 1: EP Memory (device-local allocate/free) │  │  │  │  │
│  │  │  │  └──────────────────────────────────────────────────┘  │  │  │  │
│  │  │  └────────────────────────────────────────────────────────┘  │  │  │
│  │  └──────────────────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

**Why five layers?**

| Layer | Scope | Question it answers |
|---|---|---|
| 1. EP Memory | One device, one allocation | "Where do these bytes live on this device?" |
| 2. Weight Residency | One session, one model | "Which weight regions are cold/warm/hot?" |
| 3a. DeviceGovernor | One device, all sessions | "How much VRAM/device memory can each session use?" |
| 3b. HostGovernor | One machine, all devices | "How much host RAM and disk can all devices share?" |
| 4. ClusterCoordinator | All machines, all nodes | "How should memory be coordinated across machines?" |

**Why the governor split?** A device's VRAM is exclusive — only that GPU uses it. But
host RAM and disk are shared across ALL devices on the same machine. With 8 GPUs, 8
independent per-device governors each managing `host_ram_limit` would fight over the
same physical RAM. The `HostGovernor` provides a single machine-wide authority for
shared resources, while each `DeviceGovernor` manages only its exclusive device memory.

Layers compose bottom-up: the cluster coordinator calls into host governors, host
governors coordinate device governors, device governors constrain residency managers,
and residency managers allocate through EPs. No layer bypasses the one below it.

> **Mapping to DESIGN.md §26.11:** The `ResourceGovernor` described in §26.11 maps to
> what is now called `DeviceGovernor` (per-device exclusive memory). The `host_ram_limit`
> and `disk_spill_limit` fields of `ResourceLimits` are delegated to the `HostGovernor`
> (per-machine shared memory). The §26.11 interfaces and semantics remain canonical;
> this document refines the ownership boundaries.

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

## 4. Layer 3a: DeviceGovernor (Per Compute Unit — Exclusive Memory)

The DeviceGovernor is the engine-level byte-budget authority for a single
accelerator's **exclusive** memory (GPU VRAM, NPU on-chip SRAM, etc.).
**CRITICAL: there is exactly one DeviceGovernor per DEVICE, not per session.**
It is shared across all sessions on that device.

> **Mapping to DESIGN.md §26.11:** The `ResourceGovernor` described in §26.11
> corresponds to what is now called `DeviceGovernor`. The §26.11 interfaces,
> reconfigurability semantics, and error contracts remain canonical; this section
> refines scope to **device-exclusive memory only**. The `host_ram_limit` and
> `disk_spill_limit` fields of `ResourceLimits` are delegated to the
> `HostGovernor` (§5).

### 4.1 DeviceGovernor Scope

The DeviceGovernor owns **only** resources exclusive to one compute unit:

| Resource | Example | Owned by |
|---|---|---|
| Accelerator VRAM | GPU HBM, NPU SRAM | DeviceGovernor |
| Host RAM (shared) | DDR for offload, staging | HostGovernor (§5) |
| Disk spill (shared) | SSD cold tier | HostGovernor (§5) |

### 4.2 User-Facing Limit Model

The device-memory limit is expressible three ways — absolute bytes, fraction, or auto:

```rust
#[derive(Debug, Clone, Copy)]
pub enum ResourceLimit {
    Bytes(u64),
    Fraction(f32),   // of detected tier capacity
    Auto,            // sane default (90% VRAM)
}
```

The `ResourceLimits` struct splits across governor layers:

```yaml
serving:
  memory:
    limits:
      # DeviceGovernor (per device — exclusive memory)
      vram_limit: "8GiB"          # or fraction or auto

      # HostGovernor (per machine — shared across all devices)
      host_ram_limit: "16GiB"
      disk_spill_limit: null
```

### 4.3 Live Reconfigurability

```rust
impl DeviceGovernor {
    pub fn set_vram_limit(&self, limit: ResourceLimit) -> Result<ReconfigureOutcome, ResourceError>;
    pub fn snapshot(&self) -> DeviceSnapshot;
}
```

Limits can change mid-session without restart. The governor holds limits behind
`ArcSwap<ResolvedLimits>` for lock-free reads on the hot admission path;
`reconfigure` serializes writers with a mutex.

### 4.4 Cross-Session Invariant

```rust
// Invariant checked on every reconfigure:
//   sum(session.max_pages or actual) ≤ budget.total_pages
//   interactive_reserve = round(reserve_fraction × budget.total_pages)
//   every per-session cap ≤ budget.total_pages − interactive_reserve
```

A single runaway session cannot blow the device's VRAM budget — all allocations go
through the same `can_allocate` gate, which the DeviceGovernor bounds in bytes.

### 4.5 Tiered Eviction on Lowering

When a VRAM limit is lowered below current usage, the DeviceGovernor drives
existing eviction tiers in order:

1. Drop **background** sessions' KV (cheap to re-prefill).
2. Offload **paused standard** sessions' KV to the warm tier — **requesting host
   RAM quota from HostGovernor** (§5) before copying.
3. Preempt **running standard** sessions (recompute from last checkpoint on resume).
4. **Interactive** sessions and `interactive_reserve` are touched last.

The call blocks until under ceiling or tiers are exhausted. If the target cannot
be met, the governor **rejects atomically**, restores the previous ceiling, and
returns `ResourceError::CannotSatisfyLoweredLimit`.

**Offload flow (DeviceGovernor → HostGovernor interaction):**

```text
GPU 0 VRAM full → DeviceGovernor: "need to evict to host"
    → HostGovernor: request_host_pages(device=GPU0, bytes=2GiB, priority=Normal)
    → HostGovernor: "host RAM has 200GB headroom, approved" → HostAllocation
    → EP: copy_async(vram → host)
```

### 4.6 VramBreakdown

The DeviceGovernor decomposes device memory usage into trackable components:

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

### 4.7 Sub-Budget Coordination: KV vs Expert LRUs

Independent KV and expert LRUs must not race for the last bytes. The DeviceGovernor
assigns coordinated sub-budgets and can rebalance them with hysteresis:

```text
VRAM ceiling = resident shared weights
             + hot expert/device cache
             + KV cache
             + activations and routing scratch
             + EP/runtime overhead
```

Both the `WeightResidencyManager` and KV cache manager receive sub-budgets from
the DeviceGovernor and return usage. On lowering a live limit: cancel speculative
reservations, evict unleased weight pages, demote KV, reduce batch/scratch, and
return an actionable minimum-working-set error if still impossible.

> See [DESIGN.md §26.11](./DESIGN.md) for the full governor design including
> config surfaces (YAML + Rust API + Python), error experience
> (`ResourceError` with what/why/how), and implementation status.

> See [DESIGN.md §43.2](./DESIGN.md) for the declaration that expert weights
> are "not KV cache" and the rationale for separate APIs with shared concepts.

### 4.8 Config Surface

**YAML** (device-specific limits in the `memory:` block):

```yaml
serving:
  memory:
    limits:
      vram_limit: "8GiB"            # absolute; or "0.9" (fraction); or "auto"
      allow_runtime_override: true   # permit live reconfigure via API
    interactive_reserve_pct: 20
    eviction_policy: priority_then_lru
```

**Rust:**

```rust
let engine = GenAiEngine::load(model, EngineConfig { limits, .. })?;
engine.device_governor(device_id).set_vram_limit(ResourceLimit::Bytes(6 << 30))?;
let snap = engine.device_governor(device_id).snapshot();
```

**Python:**

```python
engine.set_vram_limit("6GiB")          # default device
snap = engine.resource_snapshot()       # dict: per-tier used / limit / headroom
```

### 4.9 Error Experience

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
    HostQuotaDenied {
        device: DeviceId,
        requested_bytes: u64,
        host_available_bytes: u64,
        suggestions: Vec<Remedy>,
    },
}
```

---

## 5. Layer 3b: HostGovernor (Per Machine — Shared Memory)

The HostGovernor is the machine-level authority for **shared** memory resources
that all devices on a single physical host contend for. There is exactly **one
HostGovernor per machine**, regardless of how many devices it has.

### 5.1 Why a Separate Governor?

Host RAM and disk are **shared across all devices** on the same machine:

- When GPU 0 offloads weights from VRAM → host RAM, it uses the same physical
  DDR that GPU 1-7 also use for offload.
- If 8 independent DeviceGovernors each manage `host_ram_limit` independently,
  each thinking it has 25% of host RAM, they collectively claim 200% and OOM.
- Pinned memory pools, DMA staging buffers, and disk spill paths are
  machine-global OS resources.

The HostGovernor provides a **single source of truth** for shared memory.

### 5.2 HostGovernor Interface

```rust
trait HostGovernor: Send + Sync {
    /// A device requests host RAM pages for offload (VRAM → host).
    fn request_host_pages(
        &self,
        device: DeviceId,
        bytes: usize,
        priority: Priority,
    ) -> Result<HostAllocation>;

    /// Release previously granted host pages.
    fn release_host_pages(&self, alloc: HostAllocation);

    /// Current host RAM limit.
    fn host_ram_limit(&self) -> ResourceLimit;

    /// Current disk spill limit (None = disabled).
    fn disk_spill_limit(&self) -> Option<ResourceLimit>;

    /// Reconfigure host RAM limit live.
    fn set_host_ram_limit(&self, limit: ResourceLimit) -> Result<ReconfigureOutcome>;

    /// Reconfigure disk spill limit live.
    fn set_disk_spill_limit(&self, limit: Option<ResourceLimit>) -> Result<ReconfigureOutcome>;

    /// Global view: per-device host RAM usage breakdown.
    fn snapshot(&self) -> HostSnapshot;
}

/// Snapshot of machine-wide shared memory usage.
pub struct HostSnapshot {
    pub host_ram_limit_bytes: u64,
    pub host_ram_used_bytes: u64,
    pub host_ram_headroom_bytes: u64,
    /// Per-device breakdown of host RAM usage.
    pub per_device_host_usage: Vec<(DeviceId, u64)>,
    pub disk_spill_limit_bytes: Option<u64>,
    pub disk_spill_used_bytes: u64,
    pub pinned_memory_bytes: u64,
}
```

### 5.3 Host Allocation Lifecycle

When a DeviceGovernor needs to offload data from device memory to host RAM:

1. **Request:** DeviceGovernor calls `host_governor.request_host_pages(device, bytes, priority)`.
2. **Arbitrate:** HostGovernor checks total host RAM usage across all devices.
   If `current_used + requested ≤ host_ram_limit`, approve immediately.
3. **Pressure:** If over budget, HostGovernor can:
   - Ask other DeviceGovernors to release their host pages (cross-device pressure).
   - Cascade to disk spill (if enabled): move cold host pages to SSD.
   - Deny the request with `HostQuotaDenied` error.
4. **Grant:** Return a `HostAllocation` handle that tracks the grant.
5. **Release:** DeviceGovernor calls `release_host_pages()` when data is
   promoted back to VRAM or no longer needed.

### 5.4 Config Surface

**YAML** (machine-wide shared limits in the `memory:` block):

```yaml
serving:
  memory:
    limits:
      # HostGovernor (per machine — shared across all devices)
      host_ram_limit: "16GiB"       # or fraction of detected host RAM; or "auto" (25%)
      disk_spill_limit: null         # null = disabled (default)
      allow_runtime_override: true
    offload_to_cpu: true             # enables warm tier offload via HostGovernor
```

**Rust:**

```rust
engine.host_governor().set_host_ram_limit(ResourceLimit::Bytes(16 << 30))?;
let snap = engine.host_governor().snapshot();
println!("Host RAM: {} / {} bytes across {} devices",
    snap.host_ram_used_bytes, snap.host_ram_limit_bytes,
    snap.per_device_host_usage.len());
```

**Python:**

```python
engine.set_host_ram_limit("16GiB")
snap = engine.host_snapshot()  # dict: used / limit / per_device_usage
```

### 5.5 Cross-Device Arbitration

With multiple devices, the HostGovernor must decide **which device gets host RAM**
when the pool is contested:

- **Priority-based:** Interactive sessions' offload requests outrank background.
- **Proportional:** Each device gets a fair share by default, but can borrow
  from idle devices.
- **Pressure cascade:** When host RAM is full, the HostGovernor can trigger
  disk spill for the coldest host-resident data across any device.

```text
8×GPU system, 256GB host RAM, host_ram_limit = 200GB:
  GPU 0: 40GB host usage (weight offload)
  GPU 1: 35GB host usage (KV offload)
  ...
  GPU 7: 25GB host usage
  Total: 180GB / 200GB → 20GB headroom

  GPU 3 requests 30GB offload → only 20GB available
  → HostGovernor: pressure GPU 0 to spill 10GB coldest pages to disk
  → or: deny with HostQuotaDenied + suggestion to raise host_ram_limit
```

---

## 6. Layer 4: ClusterCoordinator (Cross-Node, genai-server)

### 6.1 When Is This Needed?

- **Single-machine, single-session:** Not needed. DeviceGovernor + HostGovernor (Layers 3a/3b) are sufficient.
- **Single-machine, multi-session:** DeviceGovernor enforces per-device budgets; HostGovernor
  arbitrates shared host RAM. No additional coordination layer is required for correctness.
- **Single-machine, cross-session optimizations:** The ClusterCoordinator (running locally)
  provides shared weight dedup, KV prefix sharing, and expert migration.
- **Multi-node distributed deployment:** The ClusterCoordinator coordinates across
  HostGovernors on different machines.

**Key clarification:** For single-machine deployments, the DeviceGovernor (§4) handles
per-device budgeting and the HostGovernor (§5) handles shared memory arbitration. The
ClusterCoordinator adds value only for cross-session *optimizations* or multi-node coordination.

### 6.2 Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                     genai-server                                 │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  ClusterCoordinator (global, cross-session)                │  │
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

### 6.3 ClusterCoordinator Interface

```rust
/// Global memory coordinator across sessions (single-machine optimizations
/// or multi-node coordination).
///
/// Sits above DeviceGovernors and HostGovernors, adjusting their budgets
/// and providing cross-session optimizations.
trait ClusterCoordinator: Send + Sync {
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
    /// Pushes adjustments down to each session's DeviceGovernor
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

### 6.4 How ClusterCoordinator Calls Down Into Governors

```text
ClusterCoordinator.rebalance()
  │
  ├── reads: device_governor[0].snapshot() → {used: 120GB, limit: 141GB, headroom: 21GB}
  ├── reads: device_governor[1].snapshot() → {used: 139GB, limit: 141GB, headroom: 2GB}
  │   └── GPU 1 under pressure!
  │
  ├── decides: GPU 1 needs 15GB for KV. GPU 0 has 21GB headroom.
  │   Migrate cold expert 742 (3GB) from GPU 1 → GPU 0.
  │   Lower GPU 1's expert sub-budget by 3GB, raise KV sub-budget.
  │
  ├── calls: device_governor[1].reconfigure({vram_kv: +3GB, vram_expert: -3GB})
  │   └── DeviceGovernor triggers tiered eviction on expert cache
  │
  └── calls: device_governor[0].reconfigure({vram_expert: +3GB})
      └── DeviceGovernor admits the migrated expert
```

The coordinator never sets a per-session limit that would violate the global ceiling.
If it tries, the DeviceGovernor rejects with `ResourceError::CannotSatisfyLoweredLimit`
and the coordinator rolls back.

### 6.5 Three Progressive Strategies

#### Strategy 1: Static Isolation

Each session gets a fixed budget. No cross-session coordination. The per-device
`DeviceGovernor` operates unchanged. Identical to running independent processes.

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

### 6.6 Cross-Node Memory Coordination

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

## 7. Communication Layer

The `Communicator` trait is the runtime-level communication abstraction for
distributed inference. It lives alongside EPs in the runtime — EPs produce
tensors; the Communicator moves them between devices. Full design in
[DISTRIBUTED_RUNTIME.md §3](./DISTRIBUTED_RUNTIME.md).

### 7.1 Core Trait

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

### 7.2 Five Backends

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

### 7.3 Communicator Supersedes DispatchTransport

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

### 7.4 Buffer Location Awareness

```rust
pub struct TransportCapability {
    pub send_from: Vec<DeviceType>,
    pub recv_into: Vec<DeviceType>,
    pub staging_device: DeviceId,  // fallback when device type unsupported
}
```

---

## 8. Heterogeneous Device Support

Because communication lives outside EP, different EP types coexist naturally.
Full design in [DISTRIBUTED_RUNTIME.md §5](./DISTRIBUTED_RUNTIME.md).

### 8.1 Format Negotiation at Boundaries

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

### 8.2 Mixing Scenarios

| Scenario | Devices | Communicator | Use Case |
|---|---|---|---|
| Multi-GPU single node | 8× H200, CUDA EP | NCCL | TP + EP for large models |
| Mac Studio cluster | 4× M3 Ultra, MLX EP | Thunderbolt | EP across Macs |
| Hybrid GPU+Mac | H200 + Mac Studio | Gloo (TCP) | Overflow to Mac for cold experts |
| NPU + GPU | NPU EP + CUDA EP | InProcess | NPU handles attention, GPU handles FFN |
| Multi-vendor GPU | ROCm EP + CUDA EP | Gloo/RDMA | Rare but architecturally possible |
| Dev/test | Multiple CPU EPs | InProcess | Verify distributed logic locally |

---

## 9. Hardware Topology Variants

Different hardware configurations require different governor topologies. The engine
selects the appropriate topology at startup based on hardware probing.

### 9.1 GovernorTopology Enum

```rust
/// Selected at engine startup based on hardware probing.
/// Upper layers (WeightResidencyManager, sessions, ParallelStrategy) are
/// topology-agnostic: they call generic request_device_memory() /
/// request_host_memory() and the topology routes correctly.
enum GovernorTopology {
    /// No accelerator. HostGovernor manages all memory.
    CpuOnly { host: Arc<HostGovernor> },
    /// Discrete device(s) with separate VRAM + shared host RAM.
    Discrete {
        host: Arc<HostGovernor>,
        devices: Vec<Arc<DeviceGovernor>>,
    },
    /// Unified memory (Apple Silicon, DGX Spark). Single governor, logical partitions.
    Unified { governor: Arc<UnifiedGovernor> },
}
```

**Key design point:** Upper layers (`WeightResidencyManager`, sessions,
`ParallelStrategy`) don't need to know which topology they're on. They call
generic `request_device_memory()` / `request_host_memory()` and the topology
routes correctly.

### 9.2 Variant 1: CPU-Only

**Example:** Inference on a server with no GPU.

- No `DeviceGovernor` — there is no exclusive device memory to manage.
- `HostGovernor` manages **all** memory (host RAM for weights, activations, KV cache).
- The warm and hot tiers collapse: everything lives in host RAM.
- Disk spill provides the cold tier.

```text
GovernorTopology::CpuOnly
└── HostGovernor (manages host RAM as both "device" and "host" memory)
    ├── host_ram_limit: "32GiB"
    └── disk_spill_limit: "100GiB"
```

### 9.3 Variant 2: Single GPU + CPU

**Example:** Desktop with one discrete GPU.

- 1 `DeviceGovernor` for the GPU's VRAM.
- 1 `HostGovernor` for host RAM offload and disk spill.
- The simplest discrete topology. No cross-device arbitration needed.

```text
GovernorTopology::Discrete
├── HostGovernor (host RAM + disk)
└── DeviceGovernor[GPU 0] (VRAM)
```

### 9.4 Variant 3: Multi-GPU Discrete

**Example:** 8×H200 server.

- N `DeviceGovernor`s, one per GPU, each managing exclusive VRAM.
- 1 `HostGovernor` arbitrating shared host RAM across all N devices.
- HostGovernor prevents 8 GPUs from collectively over-committing host RAM.
- `ClusterCoordinator` optional for cross-session optimizations (weight dedup,
  expert migration).

```text
GovernorTopology::Discrete
├── HostGovernor (256GB DDR shared across all GPUs)
├── DeviceGovernor[GPU 0] (141GB HBM)
├── DeviceGovernor[GPU 1] (141GB HBM)
│   ...
└── DeviceGovernor[GPU 7] (141GB HBM)
```

### 9.5 Variant 4: GPU + NPU

**Example:** Intel Core Ultra with Arc GPU + NPU, or Qualcomm with Adreno GPU + Hexagon NPU.

- Each accelerator gets its own `DeviceGovernor`.
- NPU device memory is typically tiny (a few MB of on-chip SRAM); it relies heavily
  on host DMA for weight streaming.
- The NPU's `DeviceGovernor` will frequently call `HostGovernor.request_host_pages()`
  for DMA pinning — these pinned pages **must not be evicted** by GPU offload pressure.
- HostGovernor needs a **pin/lease mechanism** to distinguish DMA-pinned pages from
  evictable offload pages.

```text
GovernorTopology::Discrete
├── HostGovernor (host RAM, with pinned vs pageable tracking)
├── DeviceGovernor[GPU] (VRAM: 8GB)
└── DeviceGovernor[NPU] (on-chip SRAM: 4MB, relies on host DMA)
```

### 9.6 Variant 5: Unified Memory (Apple Silicon, DGX Spark)

**Example:** M4 Ultra Mac Studio, NVIDIA DGX Spark (Grace Blackwell).

On unified memory architectures, "device memory" and "host memory" are the **same
physical DRAM**. The GPU/NPU and CPU share a single memory pool with hardware
coherence. Separate DeviceGovernor and HostGovernor would create a false dichotomy.

- `UnifiedGovernor` replaces both DeviceGovernor and HostGovernor.
- Manages **logical partitions** within unified memory:
  - Device working set (what the GPU/NPU is actively using)
  - Host working set (what the CPU is actively using)
  - Shared weight pages (accessible by both without copying)
- No copy between "host" and "device" — just pointer sharing.
- Apple's `recommendedMaxWorkingSetSize` provides the device partition hint.

```text
GovernorTopology::Unified
└── UnifiedGovernor (192GB unified pool)
    ├── device_partition: 160GB (GPU working set)
    ├── host_partition: 24GB (CPU working set)
    └── shared: 8GB (weights readable by both, no copy)
```

### 9.7 Variant 6: Multi-Node Cluster

**Example:** 4× Mac Studio cluster via Thunderbolt 5, or multi-node DGX.

- Each node runs its own governor topology (any of variants 1–5).
- `ClusterCoordinator` sits above per-node `HostGovernor`s.
- Cross-node expert migration and prefix sharing via `Communicator` (§7).

```text
┌─ Node 0 ───────────────────┐    ┌─ Node 1 ───────────────────┐
│ UnifiedGovernor (M4 Ultra)    │    │ Discrete (8×H200)           │
│ └── 192GB unified             │    │ ├── HostGovernor (256GB DDR) │
│                               │    │ └── 8× DeviceGovernor       │
└───────────────┬───────────────┘    └───────────────┬───────────────┘
                │                                │
                └──────── ClusterCoordinator ────────┘
                           (in genai-server)
```

### 9.8 Topology-Agnostic Upper Layers

The `GovernorTopology` enum exposes a uniform interface so upper layers remain
topology-unaware:

```rust
impl GovernorTopology {
    /// Request device memory (routes to DeviceGovernor or UnifiedGovernor).
    fn request_device_memory(
        &self,
        device: DeviceId,
        bytes: usize,
    ) -> Result<DeviceAllocation>;

    /// Request host memory (routes to HostGovernor or UnifiedGovernor).
    fn request_host_memory(
        &self,
        device: DeviceId,
        bytes: usize,
        priority: Priority,
    ) -> Result<HostAllocation>;

    /// Combined snapshot across all governors.
    fn snapshot(&self) -> TopologySnapshot;
}
```

This means `WeightResidencyManager`, session scheduling, and `ParallelStrategy`
never branch on hardware type — they call the same methods regardless of whether
the system is CPU-only, discrete multi-GPU, unified, or a heterogeneous mix.

---

## 10. Decision Log

Key architectural decisions and their rationale:

### D1: Governor splits into DeviceGovernor (per device) and HostGovernor (per machine)

**Decision:** One `DeviceGovernor` per physical device manages exclusive device memory
(VRAM). One `HostGovernor` per machine manages shared host RAM and disk spill.

**Rationale:** A per-session governor cannot enforce `sum(session.usage) ≤ device_capacity`.
The DeviceGovernor is the single source of truth for device byte budgets. Host RAM and
disk are shared across all devices on a machine; per-device governors managing these
resources independently would contend over the same physical memory. The HostGovernor
provides a single machine-wide authority for shared resources.

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

### D5: Single-machine uses DeviceGovernor + HostGovernor; ClusterCoordinator only for multi-node

**Decision:** For single-machine deployments, the per-device `DeviceGovernor` manages
device memory and the `HostGovernor` arbitrates shared host resources. The
`ClusterCoordinator` adds value only for cross-session optimizations (shared weight
dedup, KV prefix sharing) or multi-node coordination.

**Rationale:** Avoid adding a coordination layer where the governor pair already
enforces all invariants. The ClusterCoordinator is an optimization layer, not a
correctness requirement for single-machine deployments.

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

### D7: Host RAM and disk are per-machine shared resources, not per-device

**Decision:** `host_ram_limit` and `disk_spill_limit` are owned by the HostGovernor
(one per machine), not by individual DeviceGovernors.

**Rationale:** Host RAM and disk are physically shared across all devices on a machine.
If each of N devices independently manages a `host_ram_limit`, they collectively risk
claiming N× the available memory. A single HostGovernor with a global view prevents
this contention and provides fair cross-device arbitration.

---

## 11. Phased Implementation

Unified across all design documents:

### Phase 1: Single-Session Weight Residency

*Maps to WEIGHT_OFFLOAD.md Phases 1-2.*

- `WeightRegionCatalog` classifies model regions (shared vs expert).
- `WeightResidencyManager` with cold/warm/hot tiers.
- `ExpertStore` facade for fused MoE kernels.
- Heat-based LRU admission for experts.
- Lease/pin lifecycle with completion fences.
- DeviceGovernor sub-budgets (KV vs expert) with hysteresis.
- DeviceGovernor is the first priority (already partially wired as `ResourceGovernor`).

### Phase 2: Governor Wiring + HostGovernor

*Maps to DESIGN.md §26.11.*

- Connect real EP/model weight usage, activation/scratch high-water marks, and
  ORT/EP allocations to the DeviceGovernor.
- `hot_expert_bytes` component in `VramBreakdown`.
- Coordinated KV + expert sub-budget rebalancing.
- Lowering-triggered live eviction (tiered: background → paused → running → interactive).
- Auto mode with real capacity detection from EP device queries.
- **HostGovernor wiring:** host RAM quota management, per-device usage tracking,
  cross-device arbitration for offload pages.
- DeviceGovernor → HostGovernor integration for VRAM eviction → host RAM offload flow.

### Phase 3: Multi-GPU Single-Node

- NCCL `Communicator` for multi-GPU collective ops.
- Shared weights via CUDA IPC (zero-copy across sessions).
- `ClusterCoordinator` Strategy 2 (shared weights + shared KV pool).
- Expert migration between GPUs based on heat.
- InProcess `Communicator` for testing.
- `GovernorTopology::Discrete` with multi-device HostGovernor arbitration.

### Phase 4: Cross-Node

- Thunderbolt 5 `Communicator` for Mac Studio cluster.
- RDMA `Communicator` for data center.
- `GlobalCoordinator` above per-node `HostGovernor`s.
- Cross-node expert migration via Communicator.
- Cross-node prefix cache lookup.

---

## 12. Open Questions

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

10. **ClusterCoordinator placement.** In genai-server process or separate daemon?

### From MoE / expert parallelism

11. **Expert-aware scheduling across sessions.** When multiple sessions share a device,
    should the governor prefer expert affinity (co-locate sessions that use
    complementary expert sets)?

12. **Prefetch speculation budget.** How many speculative prefetch bytes before the
    cost of wrong predictions exceeds the benefit?

### From governor split / topology (§4, §5, §9)

13. **HostGovernor pinned vs pageable allocation.** Should HostGovernor allocate pinned
    vs pageable host memory separately? Pinned memory is a limited OS resource.

14. **Unified memory working set size.** Unified memory devices (Apple Silicon, DGX Spark):
    DeviceGovernor and HostGovernor collapse into one — the device IS the host. How to
    define `recommendedMaxWorkingSetSize` equivalent? Apple reports it; NVIDIA unified
    devices may not.

15. **NPU DMA pinning.** HostGovernor needs a pin/lease mechanism so host pages being
    DMA'd by NPU can't be evicted by GPU offload pressure. How to track and
    enforce DMA pin lifetimes?

16. **CPU-only mode.** Should we still instantiate a DeviceGovernor for CPU (treating
    host RAM as device memory) for API uniformity, or skip it and route everything
    through HostGovernor?

---

## 13. References

- [DESIGN.md §26.11](./DESIGN.md) — Resource Governor: canonical design (stays in place)
- [DESIGN.md §43.2](./DESIGN.md) — MoE Expert Weights: "not KV cache" declaration
- [WEIGHT_OFFLOAD.md](./WEIGHT_OFFLOAD.md) — Three-tier weight residency (redirects here for §4)
- [MOE_SUPPORT.md](./MOE_SUPPORT.md) — First-class MoE support (redirects here for §7)
- [MOE_EXPERT_PARALLELISM.md](./MOE_EXPERT_PARALLELISM.md) — Session-per-GPU MoE architecture (DispatchTransport deprecated)
- [DISTRIBUTED_RUNTIME.md](./DISTRIBUTED_RUNTIME.md) — Communicator abstraction & multi-device inference
- [SCHEDULING.md](./SCHEDULING.md) — Adaptive scheduling, EP negotiation protocol
- `crates/onnx-runtime-ep-api/src/provider.rs` — ExecutionProvider trait
- `crates/onnx-genai-scheduler/src/governor.rs` — DeviceGovernor implementation (originally ResourceGovernor)
