# Distributed Runtime: Communicator Abstraction & Multi-Device Inference

> Companion to [MOE_EXPERT_PARALLELISM.md](./MOE_EXPERT_PARALLELISM.md),
> [HETEROGENEOUS_PLACEMENT.md](./HETEROGENEOUS_PLACEMENT.md),
> [SCHEDULING.md](./SCHEDULING.md), and [DESIGN.md](./DESIGN.md).
>
> Generalizes the MoE-specific `DispatchTransport` trait into a runtime-level
> `Communicator` abstraction that enables arbitrary distributed inference
> strategies across heterogeneous devices.

**Status:** Design Proposal
**Author:** Claw (with Justin)
**Date:** 2026-07-19

---

## Table of Contents

1. [Motivation](#1-motivation)
2. [Why Communication Lives Outside the EP](#2-why-communication-lives-outside-the-ep)
3. [Communicator Abstraction](#3-communicator-abstraction)
4. [Communicator Backends](#4-communicator-backends)
5. [Heterogeneous Device Support](#5-heterogeneous-device-support)
6. [Parallel Strategy Layer](#6-parallel-strategy-layer)
7. [Graph Partitioner Extensions](#7-graph-partitioner-extensions)
8. [Distributed Execution Plan](#8-distributed-execution-plan)
9. [Integration with MoE Expert Parallelism](#9-integration-with-moe-expert-parallelism)
10. [Mac Studio Cluster as First-Class Target](#10-mac-studio-cluster-as-first-class-target)
11. [Phased Implementation](#11-phased-implementation)
12. [Cross-Session Memory Coordination](#12-cross-session-memory-coordination)
13. [Open Questions](#13-open-questions)
14. [References](#14-references)

---

## 1. Motivation

The existing design documents address specific multi-device scenarios in isolation:

- **MOE_EXPERT_PARALLELISM.md** — expert-parallel dispatch with `DispatchTransport`
  (send/recv/all_reduce/all_to_all) scoped to MoE token routing.
- **HETEROGENEOUS_PLACEMENT.md** — CPU+CUDA fallback for unsupported ops on a single
  machine (currently ON HOLD).
- **SCHEDULING.md §8** — EP negotiation protocol for single-session, single-device.

What's missing is a **unified runtime-level communication layer** that enables:

1. Tensor parallelism across GPUs (attention head splitting)
2. Expert parallelism across nodes (MoE dispatch)
3. Pipeline parallelism across devices (layer-range assignment)
4. Heterogeneous EP mixing (CUDA + MLX in the same distributed graph)
5. Topology-aware placement (NVLink vs TB5 vs ethernet cost modeling)

This document defines that layer: the `Communicator` trait and the parallel strategy
abstractions built on top of it.

---

## 2. Why Communication Lives Outside the EP

**Core design decision: The Runtime orchestrates communication. EPs just compute.**

### The alternative (and why it's wrong)

One could embed NCCL calls inside the CUDA EP, gloo calls inside the CPU EP, etc.
This is how most frameworks work (Megatron-LM, DeepSpeed). It fails for us because:

1. **Heterogeneous mixing is impossible.** If CUDA EP owns NCCL and MLX EP owns its
   own transport, who mediates a CUDA→MLX transfer? A third EP? The runtime is the
   only entity that sees both.

2. **EP contract violation.** The `ExecutionProvider` trait (§4.1, `provider.rs`) is
   designed for **local computation**: `supports_op`, `get_kernel`, `allocate`,
   `deallocate`, `copy`. Adding collective ops would bloat the trait and force every
   EP (including CPU) to implement distributed primitives it doesn't need.

3. **Strategy coupling.** Embedding communication in EP ties the parallelism strategy
   (TP vs EP vs PP) to the compute backend. Switching from tensor parallelism to
   expert parallelism shouldn't require rewriting EP code.

4. **Testability.** An `InProcessCommunicator` can simulate multi-device execution
   in a single process with CPU tensors. If communication is inside EP, you need
   mock EPs — which defeats the purpose of testing.

### The right separation

```
┌──────────────────────────────────────────────────────────────────┐
│  RUNTIME (Rust, async)                                            │
│                                                                   │
│  ┌─────────────────────┐    ┌──────────────────────────────────┐ │
│  │  Parallel Strategy   │    │  Communicator                    │ │
│  │  (TP / EP / PP)      │───▶│  (all_reduce, all_to_all, ...)  │ │
│  └──────────┬──────────┘    └──────────────────────────────────┘ │
│             │                                                     │
│  ┌──────────▼──────────────────────────────────────────────────┐ │
│  │  Distributed Execution Plan                                  │ │
│  │  (compiled sequence of EP.execute + comm.collective calls)   │ │
│  └──────────┬──────────┬──────────┬──────────┬─────────────────┘ │
│             │          │          │          │                    │
│        ┌────▼────┐┌────▼────┐┌────▼────┐┌────▼────┐             │
│        │ CUDA EP ││ CUDA EP ││ MLX EP  ││ CPU EP  │             │
│        │ (GPU 0) ││ (GPU 1) ││ (Mac 0) ││ (fallbk)│             │
│        └─────────┘└─────────┘└─────────┘└─────────┘             │
└──────────────────────────────────────────────────────────────────┘
```

EPs are compute-only. The Communicator is a peer abstraction at the same runtime
level. The Parallel Strategy layer compiles a plan that interleaves EP execution
with Communicator collectives.

---

## 3. Communicator Abstraction

### 3.1 Core Trait

```rust
/// Runtime-level communication abstraction for distributed inference.
///
/// Each participant in a distributed execution group holds a `Communicator`
/// handle. The communicator manages point-to-point and collective operations
/// across devices, nodes, or processes.
///
/// # Relationship to EP
///
/// A `Communicator` is NOT part of the EP trait. It lives alongside EPs in the
/// runtime. EPs produce tensors; the Communicator moves them between devices.
/// The runtime's execution plan interleaves EP.execute() and comm.collective()
/// calls.
#[async_trait]
pub trait Communicator: Send + Sync {
    // ── Identity ──

    /// This participant's rank in the communication group.
    fn rank(&self) -> Rank;

    /// Total number of participants.
    fn world_size(&self) -> usize;

    /// Human-readable backend name (e.g., "nccl", "gloo", "thunderbolt").
    fn backend_name(&self) -> &str;

    // ── Collective operations ──

    /// In-place all-reduce: every rank ends with the element-wise reduction
    /// of all inputs. Default op is sum.
    async fn all_reduce(
        &self,
        tensor: &mut DeviceBuffer,
        len: usize,
        dtype: DType,
        op: ReduceOp,
    ) -> Result<()>;

    /// All-to-all: each rank sends a distinct chunk to every other rank and
    /// receives a distinct chunk from every other rank.
    ///
    /// `send_bufs[i]` is sent to rank `i`; `recv_bufs[i]` receives from rank `i`.
    async fn all_to_all(
        &self,
        send_bufs: &[&DeviceBuffer],
        recv_bufs: &mut [&mut DeviceBuffer],
        chunk_sizes: &[usize],
        dtype: DType,
    ) -> Result<()>;

    /// All-gather: each rank contributes a chunk; every rank receives the
    /// concatenation of all chunks.
    async fn all_gather(
        &self,
        send_buf: &DeviceBuffer,
        recv_buf: &mut DeviceBuffer,
        count: usize,
        dtype: DType,
    ) -> Result<()>;

    /// Broadcast: rank `root` sends; all other ranks receive.
    async fn broadcast(
        &self,
        buffer: &mut DeviceBuffer,
        len: usize,
        dtype: DType,
        root: Rank,
    ) -> Result<()>;

    /// Reduce-scatter: reduce + scatter in one step. Each rank ends with
    /// 1/world_size of the reduced result.
    async fn reduce_scatter(
        &self,
        send_buf: &DeviceBuffer,
        recv_buf: &mut DeviceBuffer,
        count: usize,
        dtype: DType,
        op: ReduceOp,
    ) -> Result<()>;

    // ── Point-to-point ──

    /// Send a buffer to a specific rank.
    async fn send(
        &self,
        buffer: &DeviceBuffer,
        len: usize,
        dtype: DType,
        dest: Rank,
    ) -> Result<()>;

    /// Receive a buffer from a specific rank.
    async fn recv(
        &self,
        buffer: &mut DeviceBuffer,
        len: usize,
        dtype: DType,
        source: Rank,
    ) -> Result<()>;

    // ── Synchronization ──

    /// Barrier: block until all ranks reach this point.
    async fn barrier(&self) -> Result<()>;
}

/// Rank within a communication group.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Rank(pub u32);

/// Reduction operation for collective reduce.
#[derive(Clone, Copy, Debug)]
pub enum ReduceOp {
    Sum,
    Product,
    Min,
    Max,
}
```

> [!IMPORTANT]
> **Review comment P0 — define asynchronous completion and buffer ownership.**
> `async Result<()>` does not say whether completion means "submitted to the
> transport", "visible on the destination stream", or "fully complete". NCCL
> returns after enqueue while device work remains asynchronous; synchronizing
> inside every future would also prevent compute/communication overlap. Return a
> `CommEvent`/completion fence that can be attached as a stream dependency, and
> define cancellation, timeout, error propagation, and buffer-reuse/free rules.
> The contract must also preserve the current `DeviceBuffer` invariant that its
> pointer is meaningful only in its owning EP/context.
>
> **Acceptance criteria:** every operation has a precise happens-before contract;
> buffers cannot be reused or freed before completion; CUDA, host-staged, and
> InProcess backends implement the same observable semantics without forcing a
> global synchronization.

> [!IMPORTANT]
> **Review comment P0 — add variable-size AllToAll for MoE.**
> One `chunk_sizes` array cannot express independent send/receive counts and
> offsets for dynamic expert routing. Add `all_to_all_v` with
> `send_counts/send_offsets` and `recv_counts/recv_offsets`, define whether counts
> are bytes or elements, and specify the count-exchange/capacity protocol.
>
> **Acceptance criteria:** a test where each source routes a different number of
> tokens to every destination completes without padding to a global maximum and
> reconstructs token order exactly.

### 3.2 Communication Groups

Not all ranks need to participate in every collective. Sub-groups enable
hybrid strategies (e.g., TP within a node, EP across nodes):

```rust
/// A subset of ranks that participate in a collective.
///
/// The full world is group 0. Sub-groups are created for strategies like
/// "TP within node" (ranks 0-3 on node A) + "EP across nodes" (rank 0
/// from each node).
pub struct CommGroup {
    pub id: CommGroupId,
    /// Ranks in this group (world-rank space).
    pub members: Vec<Rank>,
}

impl dyn Communicator {
    /// Create a sub-communicator scoped to `group`.
    /// Returns a new Communicator whose rank() and world_size() are
    /// relative to the group.
    fn sub_group(&self, group: &CommGroup) -> Result<Box<dyn Communicator>>;
}
```

> [!IMPORTANT]
> **Review comment P0 — make subgroup creation globally ordered.**
> Communicator creation can itself require participation by all relevant ranks.
> Arbitrary synchronous `sub_group()` calls can deadlock when ranks create groups
> in different orders. Compile all groups ahead of execution using stable group
> IDs, membership validation, and one globally deterministic creation sequence.
>
> **Acceptance criteria:** every rank derives the same group table and epoch;
> duplicate/overlapping groups are deterministic; a rank cannot lazily create a
> subgroup while another rank is already entering a collective.

### 3.3 Buffer Location Awareness

The Communicator must handle tensors on different device types. Each backend
knows which device memories it can access directly:

```rust
/// Describes what a Communicator backend can access.
pub struct TransportCapability {
    /// Device types this backend can send FROM directly.
    pub send_from: Vec<DeviceType>,
    /// Device types this backend can receive INTO directly.
    pub recv_into: Vec<DeviceType>,
    /// If a device type is not in the above lists, the runtime must stage
    /// through a supported device (e.g., host memory).
    pub staging_device: DeviceId,
}
```

For example:
- `NcclCommunicator`: sends/receives CUDA buffers directly; CPU buffers must
  stage through pinned host memory.
- `GlooCommunicator`: sends/receives CPU buffers directly; GPU buffers must
  download to host first.
- `ThunderboltCommunicator`: operates on host-accessible unified memory (MLX);
  CUDA buffers must stage.

---

## 4. Communicator Backends

### 4.1 Backend Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Communicator Backends                          │
├──────────────────────┬──────────────┬───────────────┬──────────────┤
│ NcclCommunicator     │ GlooComm     │ ThunderboltCm │ InProcessCm  │
│                      │              │               │              │
│ Multi-GPU, NVLink    │ CPU + TCP    │ Mac Studio    │ Testing /    │
│ PCIe, NVSwitch       │ ethernet     │ TB5 RDMA      │ simulation   │
│                      │              │               │              │
│ 900 GB/s (NVLink)    │ 1-25 GB/s    │ ~12 GB/s      │ memcpy       │
│ <1 μs latency        │ ~100 μs      │ ~5 μs         │ ~0 μs        │
├──────────────────────┴──────────────┴───────────────┴──────────────┤
│ RdmaCommunicator     │                                             │
│ InfiniBand / RoCE    │  (Data center cross-node, 200-400 Gbps)     │
└──────────────────────┴─────────────────────────────────────────────┘
```

### 4.2 NcclCommunicator

```rust
/// NCCL-backed communicator for NVIDIA multi-GPU.
///
/// Operates directly on CUDA device buffers. Leverages NVLink for
/// intra-node and InfiniBand/RoCE for inter-node when available.
pub struct NcclCommunicator {
    /// NCCL communicator handle (from ncclCommInitRank).
    comm: NcclComm,
    /// CUDA stream dedicated to communication (overlaps with compute).
    stream: CudaStream,
    rank: Rank,
    world_size: usize,
}

impl NcclCommunicator {
    /// Initialize from a unique ID shared across all ranks.
    ///
    /// Rank 0 generates the ID; other ranks receive it via a rendezvous
    /// mechanism (e.g., shared file, TCP socket, environment variable).
    pub fn new(unique_id: NcclUniqueId, rank: Rank, world_size: usize) -> Result<Self>;
}
```

### 4.3 GlooCommunicator

```rust
/// Gloo-backed communicator for CPU tensors over TCP/IP.
///
/// Suitable for CPU-only inference, development, or as a fallback when
/// no specialized transport is available. Also used for control-plane
/// coordination (e.g., broadcasting the dispatch plan to all nodes).
pub struct GlooCommunicator {
    context: GlooContext,
    rank: Rank,
    world_size: usize,
}
```

### 4.4 ThunderboltCommunicator

```rust
/// Thunderbolt 5 communicator for Mac Studio clusters.
///
/// Uses RDMA semantics over TB5 (~12 GB/s bilateral per link).
/// Operates on host-accessible unified memory (MLX tensors live in
/// unified memory on Apple Silicon, so no staging is needed).
///
/// TB5 topology is typically daisy-chain or star through a hub.
/// The communicator discovers topology at init and optimizes collective
/// patterns accordingly (e.g., ring all-reduce along the chain).
pub struct ThunderboltCommunicator {
    /// Discovered TB5 peer connections.
    peers: Vec<TbPeerConnection>,
    /// Topology graph for optimal collective routing.
    topology: TbTopology,
    rank: Rank,
    world_size: usize,
}

/// TB5 topology types affect collective algorithm selection.
pub enum TbTopology {
    /// Direct daisy-chain: Mac0 ↔ Mac1 ↔ Mac2 ↔ Mac3
    /// Best for ring all-reduce.
    DaisyChain { order: Vec<Rank> },
    /// Star through a TB5 hub: all nodes connect to a central switch.
    /// Best for tree-based collectives.
    Star { hub_id: String },
    /// Arbitrary — fall back to pairwise send/recv.
    Mesh,
}
```

### 4.5 RdmaCommunicator

```rust
/// InfiniBand / RoCE RDMA communicator for data-center cross-node.
///
/// 200-400 Gbps per link. Supports GPUDirect RDMA for direct GPU↔NIC
/// transfers without host staging.
pub struct RdmaCommunicator {
    /// ibverbs queue pairs, one per peer.
    qps: Vec<IbvQueuePair>,
    rank: Rank,
    world_size: usize,
    /// Whether GPUDirect RDMA is available (CUDA buffers → NIC directly).
    gpu_direct: bool,
}
```

### 4.6 InProcessCommunicator

```rust
/// In-process communicator for testing and simulation.
///
/// All "ranks" live in the same process, sharing memory. Collectives
/// are implemented as simple memcpy between buffers. Useful for:
/// - Unit testing parallel strategies without real multi-GPU
/// - Verifying execution plan correctness
/// - Profiling the orchestration overhead in isolation
pub struct InProcessCommunicator {
    /// Shared state across all simulated ranks.
    shared: Arc<InProcessSharedState>,
    rank: Rank,
    world_size: usize,
}

struct InProcessSharedState {
    /// Barrier counter + condvar.
    barrier: (Mutex<usize>, Condvar),
    /// Mailboxes for point-to-point: mailbox[src][dst] = Option<Vec<u8>>
    mailboxes: Vec<Vec<Mutex<Option<Vec<u8>>>>>,
}
```

---

## 5. Heterogeneous Device Support

### 5.1 The Key Insight

Because communication lives outside EP, different EP types coexist naturally
in the same distributed graph. The Communicator is the bridge:

```
┌───────────────┐         ┌───────────────┐
│   CUDA EP     │         │   MLX EP      │
│   (GPU 0)     │         │   (Mac 0)     │
│               │         │               │
│  Produces:    │         │  Produces:    │
│  CUDA buffer  │         │  MLX unified  │
│  (device mem) │         │  (host mem)   │
└───────┬───────┘         └───────┬───────┘
        │                         │
        ▼                         ▼
┌───────────────────────────────────────────┐
│           Communicator                     │
│                                           │
│  1. Download CUDA buffer → pinned host    │
│  2. Transfer host → host (TCP / TB5)      │
│  3. Target EP reads from host buffer      │
│                                           │
│  (Or: GPUDirect RDMA if both are CUDA)    │
└───────────────────────────────────────────┘
```

### 5.2 Format Negotiation at Boundaries

Different EPs may use different tensor layouts:

```rust
/// Tensor format descriptor for cross-EP communication.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorFormat {
    pub dtype: DType,
    pub layout: TensorLayout,      // Contiguous, ChannelLast, etc.
    pub quantization: Option<QuantFormat>,  // FP4, INT8, etc.
}

/// Inserted by the runtime at EP boundaries when formats differ.
pub struct FormatConverter {
    pub source: TensorFormat,
    pub target: TensorFormat,
    /// Which device to run the conversion on (prefer the faster one).
    pub convert_on: DeviceId,
}
```

The runtime inserts format conversion nodes at boundaries automatically.
For example, CUDA EP producing row-major FP16 → MLX EP expecting column-major
FP16: the converter transposes on whichever device is cheaper.

> [!IMPORTANT]
> **Review comment P2 — complete the boundary format contract.**
> `TensorFormat` needs concrete shape/strides, logical and wire dtype,
> quantization parameters (scale, zero point, block layout/version), alignment,
> and ownership/lifetime information. Conversion must be selected and compiled
> into the immutable execution plan, not inserted dynamically after plan freeze.
>
> **Acceptance criteria:** a boundary tensor is self-describing enough for either
> peer to validate allocation size and layout; conversion workspace is budgeted;
> unsupported conversions fail at plan compilation.

### 5.3 Heterogeneous Mixing Scenarios

| Scenario | Devices | Communicator | Use Case |
|---|---|---|---|
| Multi-GPU single node | 8× H200, CUDA EP | NCCL | TP + EP for large models |
| Mac Studio cluster | 4× M3 Ultra, MLX EP | Thunderbolt | EP across Macs |
| Hybrid GPU+Mac | H200 node + Mac Studio | Gloo (TCP) | Overflow to Mac for cold experts |
| NPU + GPU | NPU EP + CUDA EP | InProcess | NPU handles attention, GPU handles FFN |
| Multi-vendor GPU | AMD ROCm EP + NVIDIA CUDA EP | Gloo/RDMA | Rare but architecturally possible |
| Dev/test | Multiple CPU EPs | InProcess | Verify distributed logic locally |

### 5.4 Device Capability Registry

```rust
/// Runtime-level registry of all devices in the distributed group.
pub struct DeviceTopology {
    /// All devices participating in this distributed session.
    pub devices: Vec<DeviceInfo>,
    /// Pairwise bandwidth between devices (bytes/sec).
    /// bandwidth[i][j] = measured or configured bandwidth from device i to j.
    pub bandwidth: Vec<Vec<u64>>,
    /// Pairwise latency between devices (nanoseconds).
    pub latency: Vec<Vec<u64>>,
}

pub struct DeviceInfo {
    pub id: DeviceId,
    pub rank: Rank,
    pub ep_type: String,           // "cuda_ep", "mlx_ep", "cpu_ep"
    pub memory_bytes: u64,         // Total device memory
    pub compute_tflops: f32,       // Peak compute throughput
    pub node_id: NodeId,           // Physical machine
}
```

---

## 6. Parallel Strategy Layer

Between the Communicator (primitive ops) and the EP (compute), the Parallel
Strategy layer defines **how** to decompose a model across devices.

```
┌─────────────────────────────────────────────────────┐
│             Parallel Strategy Layer                  │
│                                                      │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ │
│  │ TensorParall │ │ ExpertParall │ │ PipelineParal│ │
│  │              │ │              │ │              │ │
│  │ Splits heads │ │ Distributes  │ │ Assigns layer│ │
│  │ and columns  │ │ MoE experts  │ │ ranges to    │ │
│  │ across ranks │ │ across ranks │ │ stages       │ │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘ │
│         │                │                │         │
│         └────────────────┼────────────────┘         │
│                          ▼                          │
│              ┌───────────────────────┐              │
│              │  HybridStrategy       │              │
│              │  (composes TP+EP+PP)  │              │
│              └───────────────────────┘              │
└─────────────────────────────────────────────────────┘
         │                              │
         ▼                              ▼
   Communicator                    EP.execute()
   (collective ops)                (local compute)
```

### 6.1 TensorParallel

Splits attention heads and FFN columns across ranks within a communication group:

```rust
/// Tensor parallelism strategy for dense layers.
///
/// Attention: each rank holds H/N heads (Q, K, V projections sliced column-wise).
/// FFN: column-parallel on the first linear, row-parallel on the second.
/// Requires AllReduce after each parallel region.
pub struct TensorParallel {
    /// Communication group for this TP region (typically intra-node).
    pub group: CommGroup,
    /// How attention heads are distributed.
    pub head_assignment: Vec<Range<usize>>,  // head_assignment[rank] = head range
    /// FFN column split boundaries.
    pub ffn_column_splits: Vec<usize>,
}

impl TensorParallel {
    /// Communication pattern per transformer block:
    ///
    /// 1. Each rank computes attention for its head shard
    /// 2. AllReduce(attention_output)  ← Communicator
    /// 3. Each rank computes FFN column shard
    /// 4. AllReduce(ffn_output)        ← Communicator
    pub fn communication_per_block(&self) -> Vec<CollectiveOp> {
        vec![
            CollectiveOp::AllReduce { group: self.group.id, tag: "attn" },
            CollectiveOp::AllReduce { group: self.group.id, tag: "ffn" },
        ]
    }
}
```

### 6.2 ExpertParallel

Distributes MoE experts across ranks. The router runs on all ranks (or a
designated coordinator); expert dispatch uses All-to-All:

```rust
/// Expert parallelism strategy for MoE layers.
///
/// Each rank holds a shard of experts. The router produces top-K expert IDs
/// per token. All-to-All sends tokens to the ranks holding their target experts,
/// then All-to-All gathers results back.
///
/// See MOE_EXPERT_PARALLELISM.md §4 for placement strategies (contiguous,
/// round-robin, affinity-aware).
pub struct ExpertParallel {
    /// Communication group for expert dispatch (can span nodes).
    pub group: CommGroup,
    /// Expert placement: expert_id → rank.
    pub placement: ExpertPlacement,
    /// Whether shared experts are replicated on all ranks.
    pub replicate_shared: bool,
}

impl ExpertParallel {
    /// Communication pattern per MoE layer:
    ///
    /// 1. AllGather(router_input) if router is not replicated
    /// 2. AllToAll(token_dispatch)     ← send tokens to expert owners
    /// 3. [local expert compute]
    /// 4. AllToAll(expert_results)     ← gather results back
    pub fn communication_per_moe_layer(&self) -> Vec<CollectiveOp> {
        vec![
            CollectiveOp::AllToAll { group: self.group.id, tag: "expert_dispatch" },
            CollectiveOp::AllToAll { group: self.group.id, tag: "expert_gather" },
        ]
    }
}
```

### 6.3 PipelineParallel

Assigns layer ranges to stages. Micro-batch scheduling fills the pipeline:

```rust
/// Pipeline parallelism: split model layers across stages (ranks).
///
/// Stage 0: layers 0..L/S
/// Stage 1: layers L/S..2L/S
/// ...
///
/// Uses micro-batch scheduling (1F1B or interleaved) to keep all stages busy.
pub struct PipelineParallel {
    /// Communication group for pipeline stages.
    pub group: CommGroup,
    /// Layer range per stage.
    pub stage_layers: Vec<Range<usize>>,
    /// Number of micro-batches to fill the pipeline.
    pub num_micro_batches: usize,
    /// Schedule type.
    pub schedule: PipelineSchedule,
}

pub enum PipelineSchedule {
    /// Simple 1-forward-1-backward (for inference: 1F1F).
    Gpipe,
    /// Interleaved: each stage holds non-contiguous layer chunks.
    /// Reduces pipeline bubble at the cost of more communication.
    Interleaved { chunks_per_stage: usize },
}

impl PipelineParallel {
    /// Communication pattern: point-to-point send/recv between adjacent stages.
    ///
    /// stage[i] --send--> stage[i+1]  (activations)
    /// stage[i+1] completes its layers
    /// stage[i+1] --send--> stage[i+2]
    pub fn communication_per_micro_batch(&self) -> Vec<CollectiveOp> {
        (0..self.stage_layers.len() - 1)
            .map(|i| CollectiveOp::SendRecv {
                sender: self.group.members[i],
                receiver: self.group.members[i + 1],
                tag: "pipeline_activation",
            })
            .collect()
    }
}
```

### 6.4 HybridStrategy

Real deployments combine strategies. For example, 8×H200 with a 2.8T MoE model:

```
┌──────────────── Node 0 (8× H200) ──────────────────┐
│                                                      │
│  TP Group 0 (NVLink)    TP Group 1 (NVLink)         │
│  ┌──────┐ ┌──────┐     ┌──────┐ ┌──────┐           │
│  │GPU 0 │ │GPU 1 │     │GPU 4 │ │GPU 5 │           │
│  │TP r0 │ │TP r1 │     │TP r0 │ │TP r1 │           │
│  │EP 0  │ │EP 1  │     │EP 4  │ │EP 5  │           │
│  └──────┘ └──────┘     └──────┘ └──────┘           │
│  ┌──────┐ ┌──────┐     ┌──────┐ ┌──────┐           │
│  │GPU 2 │ │GPU 3 │     │GPU 6 │ │GPU 7 │           │
│  │TP r2 │ │TP r3 │     │TP r2 │ │TP r3 │           │
│  │EP 2  │ │EP 3  │     │EP 6  │ │EP 7  │           │
│  └──────┘ └──────┘     └──────┘ └──────┘           │
│                                                      │
│  TP: AllReduce within TP group (4 GPUs, NVLink)     │
│  EP: AllToAll across all 8 GPUs (NVLink/NVSwitch)   │
└──────────────────────────────────────────────────────┘
```

```rust
/// Composed parallelism strategy.
pub struct HybridStrategy {
    /// Tensor parallelism groups (intra-node, high bandwidth).
    pub tp: Option<TensorParallel>,
    /// Expert parallelism groups (can span nodes).
    pub ep: Option<ExpertParallel>,
    /// Pipeline parallelism stages (across nodes).
    pub pp: Option<PipelineParallel>,
}

impl HybridStrategy {
    /// Typical configuration: TP within node, EP across nodes.
    pub fn tp_intra_ep_inter(
        tp_size: usize,
        ep_size: usize,
        topology: &DeviceTopology,
    ) -> Self;

    /// Validate that strategy groups are non-overlapping and cover all ranks.
    pub fn validate(&self) -> Result<()>;
}
```

---

## 7. Graph Partitioner Extensions

### 7.1 Extending the ILP for Distributed Placement

The existing graph partitioner (SCHEDULING.md §8, DESIGN.md) uses ILP to assign
nodes to EPs on a single device. For distributed inference, the ILP extends with:

1. **Device set** — placement candidates are (EP, device) pairs, not just EPs.
2. **Communication cost** — edges crossing device boundaries incur transfer cost.
3. **Topology-aware bandwidth** — the cost depends on which devices are involved.

```rust
/// Extended cost model for distributed placement.
pub struct DistributedCostModel {
    /// Base cost model (compute cost per node per EP).
    pub compute: Box<dyn ComputeCostModel>,
    /// Topology for bandwidth/latency between devices.
    pub topology: DeviceTopology,
}

impl DistributedCostModel {
    /// Communication cost of an edge between two devices.
    ///
    /// cost = tensor_size_bytes / bandwidth[src][dst] + latency[src][dst]
    ///
    /// This becomes an edge weight in the ILP objective.
    pub fn comm_cost(
        &self,
        tensor_bytes: usize,
        src_device: DeviceId,
        dst_device: DeviceId,
    ) -> f64 {
        let bw = self.topology.bandwidth[src_device.index()][dst_device.index()] as f64;
        let lat = self.topology.latency[src_device.index()][dst_device.index()] as f64;
        if bw == 0.0 {
            return f64::INFINITY; // Devices cannot communicate
        }
        (tensor_bytes as f64 / bw) + (lat / 1e9) // seconds
    }
}
```

> [!IMPORTANT]
> **Review comment P1 — introduce globally stable device identity.**
> The existing `DeviceId` is a device type plus a local ordinal and cannot be
> used directly as a cross-node matrix index; local ordinals repeat on every
> node. Introduce `GlobalDeviceId { node, local_device }` (or equivalent) and an
> explicit dense topology-index map. Extend the cost model to account for
> direction, staging/conversion, shared-link contention, and collective
> algorithm rather than treating every transfer as an isolated point-to-point
> edge.
>
> **Acceptance criteria:** two nodes may both contain `CUDA:0` without identity
> collision, and all matrix lookups are validated through the topology map.

### 7.2 Bandwidth Reference

| Interconnect | Bandwidth | Typical Latency | Scenario |
|---|---|---|---|
| NVLink (H100/H200) | 900 GB/s | <1 μs | Intra-node GPU↔GPU |
| NVSwitch (DGX) | 900 GB/s (all-to-all) | ~1 μs | Full bisection bandwidth |
| PCIe 5.0 x16 | 64 GB/s | ~5 μs | GPU↔host, GPU↔GPU (no NVLink) |
| Thunderbolt 5 | ~12 GB/s | ~5 μs | Mac Studio↔Mac Studio |
| InfiniBand HDR | 25 GB/s | ~1 μs | Data center cross-node |
| InfiniBand NDR | 50 GB/s | ~1 μs | Data center cross-node |
| 100GbE | 12.5 GB/s | ~10 μs | Ethernet cross-node |
| 10GbE | 1.25 GB/s | ~50 μs | Commodity ethernet |

### 7.3 Placement Constraints

The ILP must respect:

```rust
/// Constraints the graph partitioner respects during distributed placement.
pub struct PlacementConstraints {
    /// Memory budget per device (bytes). Placement must not exceed.
    pub memory_budgets: Vec<u64>,
    /// Nodes that MUST be on the same device (e.g., attention Q/K/V).
    pub colocate: Vec<Vec<NodeId>>,
    /// Nodes that MUST be on a specific device (e.g., EP claims).
    pub pin: Vec<(NodeId, DeviceId)>,
    /// Maximum allowed communication volume (bytes) across a boundary
    /// class (e.g., "cross-node" < 1 GB per step).
    pub max_cross_boundary_bytes: Option<u64>,
}
```

---

## 8. Distributed Execution Plan

### 8.1 Compilation

The runtime compiles a distributed execution plan from the model graph,
parallel strategy, and device topology:

```rust
/// A compiled distributed execution plan.
///
/// The plan is a sequence of steps executed by the runtime. Each step is
/// either a local EP execution or a communication collective. The runtime
/// executes steps in order, with parallelism expressed by concurrent steps
/// that are independent.
pub struct DistributedPlan {
    /// Ordered steps. Steps within the same `stage` may execute concurrently.
    pub steps: Vec<PlanStep>,
    /// Per-rank subplans (each rank only executes its own steps).
    pub rank_plans: Vec<RankPlan>,
}

pub enum PlanStep {
    /// Execute a subgraph on a specific EP.
    Compute {
        rank: Rank,
        ep: EpId,
        subgraph: SubgraphId,
        inputs: Vec<TensorRef>,
        outputs: Vec<TensorRef>,
    },
    /// Collective communication.
    Collective {
        op: CollectiveOp,
        inputs: Vec<TensorRef>,
        outputs: Vec<TensorRef>,
    },
    /// Format conversion at EP boundary.
    Convert {
        rank: Rank,
        converter: FormatConverter,
        input: TensorRef,
        output: TensorRef,
    },
}

/// Per-rank view of the distributed plan.
pub struct RankPlan {
    pub rank: Rank,
    pub ep: EpId,
    pub device: DeviceId,
    /// This rank's steps (indexes into DistributedPlan.steps).
    pub step_indices: Vec<usize>,
}
```

> [!IMPORTANT]
> **Review comment P0 — represent dependencies and collective ordering in the plan.**
> The text promises concurrent steps within a `stage`, but the data model has no
> stage, dependency edges, stream, completion event, or collective sequence.
> Replace the ordered list convention with a DAG (or equivalent explicit
> dependency model) containing `StepId`, rank/group, stream, buffer liveness, and
> a stable per-group collective sequence.
>
> **Acceptance criteria:** plan compilation proves that every rank in a group
> submits an identical ordered collective signature; independent compute can
> overlap communication; buffer deallocation follows the final completion event.

### 8.2 Example: TP Attention + EP MoE (One Transformer Block)

For a 4-GPU setup with 2-way TP and 4-way EP:

```
Step  Rank 0 (GPU 0)         Rank 1 (GPU 1)         Rank 2 (GPU 2)         Rank 3 (GPU 3)
─────┬──────────────────────┬──────────────────────┬──────────────────────┬──────────────────────
  1  │ EP.exec(attn_shard0) │ EP.exec(attn_shard1) │ EP.exec(attn_shard0) │ EP.exec(attn_shard1)
     │ [heads 0..H/2]       │ [heads H/2..H]       │ [heads 0..H/2]       │ [heads H/2..H]
─────┤                      │                      │                      │
  2  │ comm.all_reduce(attn_out, tp_group_0)       │ comm.all_reduce(attn_out, tp_group_1)
     │ [TP groups: {0,1}, {2,3}]                   │                      │
─────┤                      │                      │                      │
  3  │ EP.exec(router)      │ EP.exec(router)      │ EP.exec(router)      │ EP.exec(router)
     │ [replicated]         │ [replicated]         │ [replicated]         │ [replicated]
─────┤                      │                      │                      │
  4  │ comm.all_to_all(token_dispatch, ep_group)   │                      │
     │ [EP group: {0,1,2,3}]                       │                      │
─────┤                      │                      │                      │
  5  │ EP.exec(experts_0)   │ EP.exec(experts_1)   │ EP.exec(experts_2)   │ EP.exec(experts_3)
     │ [experts 0..E/4]     │ [experts E/4..E/2]   │ [experts E/2..3E/4]  │ [experts 3E/4..E]
─────┤                      │                      │                      │
  6  │ comm.all_to_all(expert_results, ep_group)   │                      │
     │                      │                      │                      │
─────┤                      │                      │                      │
  7  │ EP.exec(combine)     │ EP.exec(combine)     │ EP.exec(combine)     │ EP.exec(combine)
     │ [weighted sum]       │ [weighted sum]       │ [weighted sum]       │ [weighted sum]
─────┴──────────────────────┴──────────────────────┴──────────────────────┴──────────────────────
```

### 8.3 Execution Engine

```rust
/// Executes a distributed plan across ranks.
pub struct DistributedExecutor {
    /// One EP per rank (or multiple if hybrid).
    eps: Vec<Arc<dyn ExecutionProvider>>,
    /// Communicator for this execution group.
    comm: Arc<dyn Communicator>,
    /// Sub-communicators for strategy groups (TP groups, EP groups).
    sub_comms: HashMap<CommGroupId, Arc<dyn Communicator>>,
    /// The compiled plan.
    plan: DistributedPlan,
}

impl DistributedExecutor {
    /// Execute one forward pass (one token or one micro-batch).
    pub async fn forward(
        &self,
        rank: Rank,
        inputs: &[Tensor],
    ) -> Result<Vec<Tensor>> {
        let rank_plan = &self.plan.rank_plans[rank.0 as usize];

        for &step_idx in &rank_plan.step_indices {
            match &self.plan.steps[step_idx] {
                PlanStep::Compute { ep, subgraph, inputs, outputs, .. } => {
                    self.eps[ep.0 as usize].execute_subgraph(subgraph, inputs, outputs)?;
                }
                PlanStep::Collective { op, inputs, outputs } => {
                    self.execute_collective(op, inputs, outputs).await?;
                }
                PlanStep::Convert { converter, input, output, .. } => {
                    converter.convert(input, output)?;
                }
            }
        }

        Ok(self.collect_outputs(rank))
    }
}
```

> [!IMPORTANT]
> **Review comment P1 — extend `FrozenPlan` instead of creating a second plan model.**
> `execute_subgraph()` is not part of the current `ExecutionProvider` trait, and
> `EpId` alone cannot represent EP instance, device, session, and expert shard.
> Reuse the accepted `FrozenPlan`/`PartitionTarget`/`PartitionId` model and execute
> compiled partition artifacts. Ensure `sub_comms` is selected by each collective
> step and make the public `inputs` argument participate in tensor binding.
>
> **Acceptance criteria:** distributed and single-device compilation share one
> immutable plan representation; no executor indexes an arbitrary EP vector with
> an unvalidated ID; every compute step references a compiled partition.

---

## 9. Integration with MoE Expert Parallelism

### 9.1 DispatchTransport → Communicator

The `DispatchTransport` trait from [MOE_EXPERT_PARALLELISM.md §8](./MOE_EXPERT_PARALLELISM.md#8-communication-primitives)
is a specialization of the `Communicator`:

```rust
// MOE_EXPERT_PARALLELISM.md defined:
pub trait DispatchTransport: Send + Sync {
    async fn send(&self, target: GpuId, data: &Tensor) -> Result<()>;
    async fn recv(&self, source: GpuId) -> Result<Tensor>;
    async fn all_reduce(&self, data: &mut Tensor) -> Result<()>;
    async fn all_to_all(&self, send_bufs: &[Tensor], recv_bufs: &mut [Tensor]) -> Result<()>;
}

// This document's Communicator is a superset:
// - send/recv      → Communicator::send / Communicator::recv
// - all_reduce     → Communicator::all_reduce
// - all_to_all     → Communicator::all_to_all
// + all_gather, broadcast, reduce_scatter, barrier, sub-groups
```

**Migration path:** `DispatchTransport` becomes a thin wrapper:

```rust
/// Adapter: wraps a Communicator into the DispatchTransport interface
/// expected by the MoE dispatch pipeline.
pub struct CommunicatorDispatchTransport {
    comm: Arc<dyn Communicator>,
    /// The communication group used for expert dispatch.
    ep_group: CommGroupId,
}

#[async_trait]
impl DispatchTransport for CommunicatorDispatchTransport {
    async fn send(&self, target: GpuId, data: &Tensor) -> Result<()> {
        self.comm.send(&data.buffer, data.len(), data.dtype, Rank(target.0)).await
    }

    async fn all_to_all(
        &self,
        send_bufs: &[Tensor],
        recv_bufs: &mut [Tensor],
    ) -> Result<()> {
        // Delegate to Communicator with the EP sub-group
        self.comm.all_to_all(/* ... */).await
    }

    // ... etc
}
```

### 9.2 Control Plane / Data Plane Preservation

The control plane / data plane separation from MOE_EXPERT_PARALLELISM.md §2
maps directly to this design:

| MoE Doc Concept | Distributed Runtime Equivalent |
|---|---|
| Control plane (expert placement, rebalancing) | `HybridStrategy` + `ExpertPlacement` |
| Data plane (NCCL collectives in GPU graph) | `DistributedPlan` steps |
| Dispatch plan table | `ExpertParallel.placement` |
| `MoeDispatchOp` (custom ONNX op) | `PlanStep::Collective(AllToAll)` |
| `ExpertSession` trait | `ExecutionProvider` + rank-specific subgraph |

### 9.3 GPU-Native Mode

For maximum performance (MOE_EXPERT_PARALLELISM.md §5.3 Mode 1), the Communicator
collectives can be **baked into the ONNX graph** as custom ops that internally call
NCCL. The `DistributedPlan` in this case is just: "fire rank 0's session, read
output." All communication happens inside the graph without returning to Rust.

The `DistributedPlan`/`DistributedExecutor` approach (explicit interleaving) is
the **orchestrated mode** (Mode 2) — more flexible, inspectable, and required
for heterogeneous EP mixing where collectives can't live inside any single EP.

> [!IMPORTANT]
> **Review comment P0 — reconcile GPU-native mode with runtime ownership.**
> Calling NCCL from an ordinary EP custom op contradicts the core decision that
> communication is runtime-owned, and a multi-rank collective cannot be launched
> by firing rank 0 alone. Describe this as lowering runtime-owned communication
> plan ops into CUDA graph-capturable calls, with every rank launching its own
> rank-local plan in the validated collective order.
>
> **Acceptance criteria:** GPU-native mode preserves the same communicator
> completion/error contract as orchestrated mode, launches all ranks, and does
> not create an undocumented communication API inside `ExecutionProvider`.

---

## 10. Mac Studio Cluster as First-Class Target

> [!IMPORTANT]
> **Review comment P1 — defer the unpublished K3 target and replace estimates with measurements.**
> K3 does not yet have a published implementation, so it cannot be a Phase 3
> acceptance target. The `12 GB/s`, `5 μs`, FP4 capacity, and `150 tokens/sec`
> figures are unvalidated upper-bound assumptions; the section also labels the
> same target both interconnect-bound and compute-bound. Keep Thunderbolt 5 RDMA
> as a supported transport hypothesis, but move this scenario to a deferred
> validation appendix and benchmark it with a released reproducible model.
>
> **Acceptance criteria:** no release milestone depends on K3; theoretical link
> rate is distinguished from measured payload/collective throughput; estimates
> include fixed collective latency, topology contention, protocol overhead,
> quantization metadata, memory bandwidth, and achieved rather than peak compute.

### 10.1 Reference Configuration

```
┌─────────────────────────────────────────────────────────────────┐
│  4× Mac Studio M3 Ultra (512 GB each) = 2 TB total             │
│  Thunderbolt 5 interconnect (~12 GB/s per link)                 │
│                                                                  │
│  Mac 0 ◄──TB5──► Mac 1 ◄──TB5──► Mac 2 ◄──TB5──► Mac 3        │
│  (orchestrator)                                                  │
│                                                                  │
│  Each Mac runs:                                                  │
│  - MLX EP (unified memory, zero-copy compute)                   │
│  - genai-server rank (Rust async runtime)                       │
│  - ThunderboltCommunicator                                      │
│                                                                  │
│  Total: 2 TB unified memory → fits K3-class 2.8T model (FP4)   │
│         + ~500 GB headroom for KV cache                         │
└─────────────────────────────────────────────────────────────────┘
```

### 10.2 Mac vs GPU: Same Abstraction, Different Trade-offs

```
                 │  8× H200 (NVLink)    │  4× Mac Studio M3 Ultra (TB5)
─────────────────┼──────────────────────┼─────────────────────────────
Memory           │  8 × 141 = 1,128 GB  │  4 × 512 = 2,048 GB
Interconnect BW  │  900 GB/s (NVLink)   │  ~12 GB/s (TB5)
Compute (FP16)   │  ~8000 TFLOPS        │  ~88 TFLOPS (4 × 22)
Communicator     │  NcclCommunicator    │  ThunderboltCommunicator
EP               │  CUDA EP             │  MLX EP
Attention        │  TP (AllReduce)      │  Replicated (no TP needed)
MoE dispatch     │  AllToAll (NCCL)     │  AllToAll (TB5)
TP needed?       │  Yes (weights > 1 GPU)│  No (full attn fits in 512 GB)
Bottleneck       │  Compute-bound       │  Interconnect-bound
```

### 10.3 Why No TP on Mac Studio

Apple Silicon unified memory means a single Mac can hold the full attention
layers alongside its expert shard. With 512 GB per node and ~350 GB of expert
weights per node (2.8T / 4 / 2 for FP4), there's ~160 GB left — more than
enough for attention weights (~15-20 GB for a 2.8T MoE's dense layers).

This eliminates AllReduce for attention entirely. Only expert All-to-All crosses
TB5 — a massive simplification:

```
Mac Studio execution per transformer block:
  1. attention_forward(full_heads)      ← local, no communication
  2. router_forward()                   ← local
  3. comm.all_to_all(expert_dispatch)   ← TB5
  4. local_experts_forward()            ← local
  5. comm.all_to_all(expert_results)    ← TB5
  6. combine()                          ← local
```

### 10.4 Latency Analysis

For a K3-class model (896 experts, top-16, 4096 hidden dim, BF16):

```
Per MoE layer All-to-All:
  Dispatch: 16 experts × batch × 4096 × 2 bytes = ~131 KB per token
  At 12 GB/s (TB5): 131 KB / 12 GB/s ≈ 11 μs
  Round-trip (dispatch + gather): ~22 μs

For 100 MoE layers:
  Total comm overhead: 100 × 22 μs = 2.2 ms per token

Compute per token (local experts):
  ~50B active params × 2 FLOPs/param = 100 GFLOPS
  At 22 TFLOPS (M3 Ultra): ~4.5 ms per token (compute-bound on Mac)

Total per-token latency: ~4.5 ms compute + ~2.2 ms comm ≈ 6.7 ms
  → ~150 tokens/sec (acceptable for interactive use)
```

The Mac Studio cluster is **compute-bound, not interconnect-bound** — the TB5
bandwidth is sufficient because expert dispatch volumes are modest.

---

## 11. Phased Implementation

### Phase 1: InProcessCommunicator + Simulated Multi-EP

**Goal:** Validate the abstraction without real hardware.

- Implement `Communicator` trait and `InProcessCommunicator`
- Implement `TensorParallel` and `ExpertParallel` strategy structs
- Implement `DistributedPlan` compilation from strategy + graph
- Test with multiple CPU EPs in one process simulating multi-device
- Verify correctness: distributed execution matches single-device results
- **No real multi-GPU or networking code.**

```rust
#[test]
fn distributed_matches_single_device() {
    let model = load_test_model();
    let single_result = run_single_device(&model, &input);

    let comm = InProcessCommunicator::new(world_size: 4);
    let strategy = ExpertParallel::contiguous(num_experts: 16, world_size: 4);
    let plan = compile_distributed_plan(&model, &strategy, &comm);
    let distributed_result = run_distributed(&plan, &input);

    assert_tensors_close(&single_result, &distributed_result, atol: 1e-5);
}
```

### Phase 2: NCCL Backend for Multi-GPU Single-Node

**Goal:** Real multi-GPU inference on a single machine.

- Implement `NcclCommunicator` wrapping NCCL2
- Integrate with existing CUDA EP
- TP for attention (AllReduce) + EP for MoE (AllToAll)
- Performance benchmarking against single-GPU baseline
- KV cache distributed across GPU shards
- **Target:** DeepSeek V3 (671B) on 8×H200

### Phase 3: Cross-Node Communication (TB5 / RDMA / gRPC)

**Goal:** Multi-machine inference.

- Implement `ThunderboltCommunicator` for Mac Studio cluster
- Implement `RdmaCommunicator` for InfiniBand
- Implement `GlooCommunicator` as TCP fallback
- Process rendezvous and rank assignment
- Fault detection and recovery (node failure)
- **Target:** K3-class 2.8T on 4× Mac Studio M3 Ultra

> [!IMPORTANT]
> **Review comment P1 — align Phase 3 with the resolved failure policy.**
> `MEMORY_ARCHITECTURE.md` resolves Phase 1–3 rank failure as abort-all and
> restart, while this phase promises fault detection and recovery. State the same
> policy here and defer partial recovery/degraded execution to Phase 4+.
>
> **Acceptance criteria:** both documents use one failure-state machine and
> define communicator abort, request failure, cleanup, and restart ownership.

### Phase 4: Heterogeneous EP Mixing

**Goal:** Different EP types in the same distributed session.

- Cross-EP format conversion at communication boundaries
- `DeviceTopology` discovery and cost modeling
- ILP partitioner with communication edge costs
- Mixed CUDA EP + MLX EP execution
- Dynamic strategy selection based on topology
- **Target:** H200 node + Mac Studio overflow for cold experts

---

## 12. Cross-Session Memory Coordination

> **Consolidated.** See [MEMORY_ARCHITECTURE.md §4-5](./MEMORY_ARCHITECTURE.md).
> Cross-session memory coordination, the `MemoryCoordinator` trait, budget
> arbitration, shared weight deduplication, and the relationship between
> governors and coordinators are consolidated there.


## 13. Open Questions

> Renumbered from §12; previous items 1-10 preserved, new items 11-13 added.

> [!IMPORTANT]
> **Review comment P1 — remove decisions already resolved elsewhere.**
> Rendezvous, failure policy, dynamic membership, backend selection, quantized
> communication phase, CUDA IPC ownership, KV pool format, and coordinator
> placement are marked resolved in `MEMORY_ARCHITECTURE.md`. Replace those entries
> with links to the canonical decisions and keep this section only for genuinely
> open communicator/execution questions such as async overlap, algorithm
> selection, speculative dispatch batching, fallback placement, and cooperative
> memory pressure.
>
> **Acceptance criteria:** no issue is simultaneously open and resolved; this
> document consistently uses `ClusterCoordinator` rather than
> `MemoryCoordinator`; each remaining open question names its decision owner and
> target phase.

1. **Rendezvous mechanism.** How do distributed ranks discover each other?
   Options: (a) environment variables like `MASTER_ADDR`/`MASTER_PORT` (PyTorch
   convention), (b) shared file on NFS, (c) built-in TCP rendezvous server in
   genai-server, (d) mDNS/Bonjour for Mac Studio cluster. Likely need multiple
   backends.

2. **Fault tolerance.** What happens when a rank crashes mid-collective?
   NCCL aborts all ranks. Do we need checkpointing, or is restart-from-scratch
   acceptable for inference? (Training needs checkpoints; inference can restart
   a request.)

3. **Dynamic rank membership.** Can ranks join/leave a live session? Useful for
   scaling up/down based on load. NCCL doesn't support this; would require
   session rebuild. Gloo/custom backends could support it.

4. **Communicator selection.** When multiple backends are available (e.g., both
   NCCL and RDMA for GPU-to-GPU across nodes), who picks? Should the runtime
   auto-select based on topology, or is it user-configured?

5. **Async overlap.** How to overlap communication with computation? NCCL supports
   CUDA stream-based overlap. For TB5/RDMA, can we overlap host-side communication
   with MLX compute? This is critical for hiding the TB5 latency.

6. **AllReduce algorithm selection.** Ring vs tree vs recursive-halving depends
   on topology and message size. Should the Communicator auto-select, or should
   the strategy layer hint?

7. **Quantized communication.** Can we reduce communication volume by sending
   FP8/INT8 and up-casting at the receiver? Lossy but could halve bandwidth
   requirements on TB5.

8. **Interaction with speculative decoding.** Multiple draft tokens route to
   different experts. Do we batch all draft token dispatches into one AllToAll,
   or dispatch per-token? Batched is more efficient but increases latency for
   the first draft.

9. **HETEROGENEOUS_PLACEMENT.md integration.** The ON HOLD single-machine
   CPU+CUDA fallback design should eventually compose with distributed placement.
   When a remote node has an unsupported op, does it fall back locally (CPU on
   that node) or route to another node? Likely local fallback first.

10. **Memory pressure across ranks.** If one rank runs out of KV cache memory,
    should it signal other ranks to evict sequences cooperatively? Or does the
    control plane manage this centrally?

11. **CUDA IPC ownership semantics.** When session 0 allocates shared weights
    and sessions 1-7 map them via IPC, who owns the allocation lifecycle? If
    session 0 crashes, all other sessions lose access. Options: (a) dedicated
    "weight server" process that outlives sessions, (b) shared mmap-backed
    allocations that survive process death, (c) accept the coupling and restart
    all sessions on any crash.

12. **KV cache sharing granularity.** Global KV pool enables prefix sharing, but
    different sessions may quantize KV differently (FP16 vs FP8). Do we enforce
    uniform KV format across sessions, or support format conversion at share
    boundaries?

13. **MemoryCoordinator placement.** Should it run in the genai-server process,
    or as a separate daemon? In-process is simpler; separate daemon survives
    server restarts but adds IPC complexity. Related to the "weight server"
    question in item 11.

---

## 14. References

- [WEIGHT_OFFLOAD.md](./WEIGHT_OFFLOAD.md) — Three-tier weight residency, `ExpertStore`, `ResourceGovernor`
- [MOE_EXPERT_PARALLELISM.md](./MOE_EXPERT_PARALLELISM.md) — Session-per-GPU MoE architecture, `DispatchTransport` trait
- [HETEROGENEOUS_PLACEMENT.md](./HETEROGENEOUS_PLACEMENT.md) — CPU+CUDA fallback placement (ON HOLD)
- [SCHEDULING.md](./SCHEDULING.md) — Adaptive scheduling, EP negotiation protocol (§8)
- [DESIGN.md](./DESIGN.md) — Project architecture, KV cache manager (§3.2), paged memory
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/) — NVIDIA Collective Communications Library
- [Gloo](https://github.com/facebookincubator/gloo) — Facebook collective communications library
- [Megatron-LM](https://arxiv.org/abs/1909.08053) — Efficient large-scale language model training with TP/PP
- [DeepSpeed](https://arxiv.org/abs/2207.00032) — ZeRO, expert parallelism, pipeline parallelism
- [Alpa](https://arxiv.org/abs/2201.12023) — Automating inter- and intra-operator parallelism (ILP-based)
- [onnx/onnx#8184](https://github.com/onnx/onnx/issues/8184) — Inference Metadata standard
