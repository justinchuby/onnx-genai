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
15. [Appendix A: Hypothetical Workloads (Pending Validation)](#appendix-a-hypothetical-workloads-pending-validation)

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
    fn rank(&self) -> RankId;

    /// Total number of participants.
    fn world_size(&self) -> usize;

    /// Human-readable backend name (e.g., "nccl", "gloo", "thunderbolt").
    fn backend_name(&self) -> &str;

    // ── Collective operations ──
    //
    // All collective and point-to-point operations return a `CommHandle`
    // representing an asynchronous completion fence. Completion semantics:
    //
    //   - "Enqueued":  the call returns `Ok(CommHandle)` once the operation
    //                  has been submitted to the transport. Device work may
    //                  still be in-flight.
    //   - "Visible":   the destination rank can observe the data (e.g., a
    //                  CUDA stream dependency has been satisfied).
    //   - "Complete":  `CommHandle::wait()` returns `Ok(())` or
    //                  `is_complete()` returns `true`. The operation is
    //                  fully done on all participating ranks.
    //
    // **Buffer reuse rule:** the caller MUST NOT reuse, free, or mutate
    // input buffers until `CommHandle::wait()` returns or `is_complete()`
    // returns `true`. The `DeviceBuffer` pointer is meaningful only in its
    // owning EP/context.
    //
    // **Cancellation:** dropping a `CommHandle` without calling `wait()` is
    // a best-effort cancel request. The transport may still complete the
    // operation.
    //
    // **Error propagation:** transport-level errors surface on `wait()`,
    // not at enqueue time. An `Err` return from the async method itself
    // indicates a synchronous failure (e.g., invalid arguments).

    /// In-place all-reduce: every rank ends with the element-wise reduction
    /// of all inputs. Default op is sum.
    async fn all_reduce(
        &self,
        tensor: &mut DeviceBuffer,
        len: usize,
        dtype: DType,
        op: ReduceOp,
    ) -> Result<CommHandle>;

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
    ) -> Result<CommHandle>;

    /// Variable-size all-to-all for MoE dynamic routing.
    ///
    /// Two-phase operation:
    /// 1. Count exchange: all ranks call `exchange_counts()` to agree on recv_counts
    /// 2. Data transfer: call `all_to_all_v()` with the agreed counts
    ///
    /// This separation ensures no rank must guess receive buffer sizes.

    /// Exchange send counts to determine recv counts for variable-size all-to-all.
    ///
    /// Each rank provides `send_counts[i]` = number of elements to send to rank `i`.
    /// Returns `recv_counts[i]` = number of elements rank `i` will send to us.
    async fn exchange_counts(
        &self,
        send_counts: &[usize],
    ) -> Result<Vec<usize>>;

    /// Variable-size all-to-all data transfer.
    ///
    /// **Validation:** The implementation asserts:
    /// - `send_counts.len() == world_size` and `recv_counts.len() == world_size`
    /// - `send_offsets.len() == world_size` and `recv_offsets.len() == world_size`
    /// - `sum(send_counts[i]) * spec.element_size() <= send_buf.len()`
    /// - `sum(recv_counts[i]) * spec.element_size() <= recv_buf.len()`
    async fn all_to_all_v(
        &self,
        send_buf: &DeviceBuffer,
        send_counts: &[usize],
        send_offsets: &[usize],
        recv_buf: &mut DeviceBuffer,
        recv_counts: &[usize],  // from exchange_counts()
        recv_offsets: &[usize],
        spec: &WireTensorSpec,  // dtype + format for byte validation
    ) -> Result<CommHandle>;

    /// All-gather: each rank contributes a chunk; every rank receives the
    /// concatenation of all chunks.
    async fn all_gather(
        &self,
        send_buf: &DeviceBuffer,
        recv_buf: &mut DeviceBuffer,
        count: usize,
        dtype: DType,
    ) -> Result<CommHandle>;

    /// Broadcast: rank `root` sends; all other ranks receive.
    async fn broadcast(
        &self,
        buffer: &mut DeviceBuffer,
        len: usize,
        dtype: DType,
        root: RankId,
    ) -> Result<CommHandle>;

    /// Reduce-scatter: reduce + scatter in one step. Each rank ends with
    /// 1/world_size of the reduced result.
    async fn reduce_scatter(
        &self,
        send_buf: &DeviceBuffer,
        recv_buf: &mut DeviceBuffer,
        count: usize,
        dtype: DType,
        op: ReduceOp,
    ) -> Result<CommHandle>;

    // ── Point-to-point ──

    /// Send a buffer to a specific rank.
    async fn send(
        &self,
        buffer: &DeviceBuffer,
        len: usize,
        dtype: DType,
        dest: RankId,
    ) -> Result<CommHandle>;

    /// Receive a buffer from a specific rank.
    async fn recv(
        &self,
        buffer: &mut DeviceBuffer,
        len: usize,
        dtype: DType,
        source: RankId,
    ) -> Result<CommHandle>;

    // ── Synchronization ──

    /// Barrier: block until all ranks reach this point.
    /// Unlike other operations, barrier is synchronous — it returns `Result<()>`
    /// (NOT `CommHandle`) because all ranks must block until convergence.
    async fn barrier(&self) -> Result<()>;
}

/// Rank within a communication group.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RankId(pub u32);

/// Reduction operation for collective reduce.
#[derive(Clone, Copy, Debug)]
pub enum ReduceOp {
    Sum,
    Product,
    Min,
    Max,
}

/// Completion handle for asynchronous communication operations.
/// Represents "enqueued to transport" — device work may still be in-flight.
pub struct CommHandle {
    inner: Box<dyn CommCompletion>,
}

pub trait CommCompletion: Send {
    /// Wait until the operation is fully visible on the destination.
    fn wait(&self) -> Result<()>;
    /// Check if complete without blocking.
    fn is_complete(&self) -> bool;
    /// Attach as a dependency on a device stream/queue.
    /// The stream will not proceed past this point until the comm is complete.
    fn attach_to_stream(&self, stream: &dyn DeviceStream) -> Result<()>;
}

/// Wire tensor specification for validating all_to_all_v transfers.
pub struct WireTensorSpec {
    pub dtype: DType,
    pub element_size: usize,
    pub format: TensorFormat,
}

impl WireTensorSpec {
    pub fn element_size(&self) -> usize { self.element_size }
}
```

### 3.2 Communication Groups

Not all ranks need to participate in every collective. Groups are compiled
from the execution plan before any collective operation begins, ensuring all
ranks create groups in the same deterministic order:

```rust
/// A subset of ranks that participate in a collective.
///
/// The full world is group 0. Sub-groups are created for strategies like
/// "TP within node" (ranks 0-3 on node A) + "EP across nodes" (rank 0
/// from each node).
pub struct CommGroup {
    pub id: GroupId,
    /// Ranks in this group (world-rank space).
    pub members: Vec<RankId>,
}

/// Communication group registry. All groups must be registered before
/// any collective operation begins. Registration order must be
/// deterministic across all ranks.
pub struct GroupRegistry {
    groups: HashMap<GroupId, Arc<dyn Communicator>>,
}

impl GroupRegistry {
    /// Register all groups in a deterministic order derived from the execution plan.
    /// Called once during plan compilation, before execution begins.
    /// All ranks must call with the same group table.
    pub fn compile(world: &dyn Communicator, plan: &[GroupSpec]) -> Result<Self>;

    /// Look up a pre-compiled sub-communicator by group ID.
    pub fn get(&self, id: GroupId) -> Option<&Arc<dyn Communicator>>;
}

pub struct GroupSpec {
    pub id: GroupId,
    pub ranks: Vec<RankId>,
}
```

Groups are **not** created lazily at runtime. The plan compiler derives all
required groups from the parallel strategy, validates membership, and compiles
them in a single globally-ordered pass. This prevents deadlocks from ranks
creating groups in different orders.

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
    pub staging_device: GlobalDeviceId,
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
    rank: RankId,
    world_size: usize,
}

impl NcclCommunicator {
    /// Initialize from a unique ID shared across all ranks.
    ///
    /// Rank 0 generates the ID; other ranks receive it via a rendezvous
    /// mechanism (e.g., shared file, TCP socket, environment variable).
    pub fn new(unique_id: NcclUniqueId, rank: RankId, world_size: usize) -> Result<Self>;
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
    rank: RankId,
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
    rank: RankId,
    world_size: usize,
}

/// TB5 topology types affect collective algorithm selection.
pub enum TbTopology {
    /// Direct daisy-chain: Mac0 ↔ Mac1 ↔ Mac2 ↔ Mac3
    /// Best for ring all-reduce.
    DaisyChain { order: Vec<RankId> },
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
    rank: RankId,
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
    rank: RankId,
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
///
/// Self-describing: either peer can validate allocation size, layout,
/// and alignment from this descriptor alone.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorFormat {
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub logical_dtype: DType,
    pub wire_dtype: DType,  // may differ if quantized for transfer
    pub quantization: Option<QuantizationParams>,
    pub alignment: usize,   // byte alignment requirement
    pub ownership: BufferOwnership,
}

pub struct QuantizationParams {
    pub scale: f64,
    pub zero_point: i64,
    pub block_size: Option<usize>,
}

pub enum BufferOwnership {
    /// Caller owns, callee borrows for the duration of the operation.
    Borrowed,
    /// Ownership transfers to the callee.
    Transferred,
}

/// Inserted by the plan compiler at EP boundaries when formats differ.
/// Conversion is selected and compiled into the immutable execution plan,
/// not inserted dynamically after plan freeze.
pub struct FormatConverter {
    pub source: TensorFormat,
    pub target: TensorFormat,
    /// Which device to run the conversion on (prefer the faster one).
    pub convert_on: GlobalDeviceId,
}
```

The plan compiler inserts format conversion steps at boundaries during
compilation. Conversion workspace is budgeted as part of the plan's memory
allocation. Unsupported conversions fail at plan compilation time.

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
    pub id: GlobalDeviceId,
    pub rank: RankId,
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
    /// Dense index map: GlobalDeviceId → matrix index.
    pub topo_index: HashMap<GlobalDeviceId, usize>,
}

/// Globally unique device identifier across a distributed cluster.
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct GlobalDeviceId {
    pub node: NodeId,
    pub local: LocalDeviceId,
}

/// Opaque node identifier assigned during rendezvous.
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct NodeId(pub u32);

/// Local device ordinal within a node (e.g., CUDA:0, CUDA:1).
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct LocalDeviceId {
    pub kind: DeviceKind,
    pub ordinal: u32,
}

impl GlobalDeviceId {
    /// Extract the local device ID for rank-local EP dispatch.
    pub fn local(&self) -> LocalDeviceId { self.local.clone() }
}

impl DistributedCostModel {
    /// Communication cost of an edge between two devices.
    ///
    /// cost = tensor_size_bytes / bandwidth[src][dst] + latency[src][dst]
    ///
    /// This becomes an edge weight in the ILP objective.
    /// All matrix lookups are validated through `topo_index`.
    pub fn comm_cost(
        &self,
        tensor_bytes: usize,
        src_device: &GlobalDeviceId,
        dst_device: &GlobalDeviceId,
    ) -> f64 {
        let src = self.topo_index[src_device];
        let dst = self.topo_index[dst_device];
        let bw = self.topology.bandwidth[src][dst] as f64;
        let lat = self.topology.latency[src][dst] as f64;
        if bw == 0.0 {
            return f64::INFINITY; // Devices cannot communicate
        }
        (tensor_bytes as f64 / bw) + (lat / 1e9) // seconds
    }
}
```

Note: Two nodes may both contain `CUDA:0` without identity collision because
`GlobalDeviceId` includes the `NodeId`. The `topo_index` map provides validated
dense indices for all bandwidth/latency matrix lookups.

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
    pub pin: Vec<(NodeId, LocalDeviceId)>,
    /// Maximum allowed communication volume (bytes) across a boundary
    /// class (e.g., "cross-node" < 1 GB per step).
    pub max_cross_boundary_bytes: Option<u64>,
}
```

---

## 8. Distributed Execution Plan

### 8.1 Compilation — Extending FrozenPlan

The runtime compiles a distributed execution plan as an **extension** of the
existing `FrozenPlan` (see SCHEDULING.md). There is ONE plan representation —
`FrozenPlan` owns compute partitions; `DistributedPlanExtension` adds
communication steps and cross-partition dependencies:

```rust
/// Distributed execution extends FrozenPlan with communication metadata.
/// There is ONE plan representation — no standalone ExecutionPlan struct.
pub struct DistributedPlanExtension {
    /// References steps in the base FrozenPlan by PartitionId.
    /// Adds communication steps between partitions.
    pub comm_steps: Vec<CommStep>,
    /// Dependency edges (both compute and comm steps share one DAG).
    pub deps: Vec<Vec<StepRef>>,
    /// Per-group collective sequence for ordering validation.
    pub collective_sequences: HashMap<GroupId, Vec<StepRef>>,
}

/// A step reference into either the base plan or comm extension.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub enum StepRef {
    /// References a FrozenPlan partition (compute step).
    Compute(PartitionId),
    /// Index into `DistributedPlanExtension::comm_steps`.
    Comm(usize),
}

pub struct CommStep {
    pub op: CollectiveOp,
    pub group: GroupId,
    pub stream: StreamId,
    pub buffer_deps: Vec<BufferId>,
}

pub type StreamId = u32;
pub type BufferId = u32;
```

The DAG structure enables:
- **Overlap:** Independent compute partitions on different streams execute
  concurrently with communication on the comm stream.
- **Correctness:** The `collective_sequences` map is verified at compile time
  to ensure all ranks in a group submit collectives in the same order.
- **Buffer safety:** A buffer's `BufferId` appears in `buffer_deps` of every
  step that reads it; deallocation is legal only after all such steps complete.
- **No duplication:** Compute steps are `PartitionId` references into the
  existing `FrozenPlan`; they are not re-defined here.

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

### 8.3 Async DAG Scheduler

The execution engine uses an async DAG scheduler that maximizes
compute/communication overlap by launching all ready steps concurrently
and waiting for ANY in-flight operation (not all):

```rust
/// Async DAG scheduler for distributed execution with maximum overlap.
pub struct DagScheduler {
    frozen_plan: Arc<FrozenPlan>,
    extension: Arc<DistributedPlanExtension>,
    in_flight: HashMap<StepRef, CommHandle>,
    completed: HashSet<StepRef>,
}

impl DagScheduler {
    /// Execute the DAG with maximum compute/communication overlap.
    pub async fn execute(&mut self, ctx: &ExecutionContext) -> Result<()> {
        loop {
            // Find all steps whose dependencies are satisfied
            let ready: Vec<StepRef> = self.all_steps()
                .filter(|s| !self.completed.contains(s))
                .filter(|s| self.deps_of(s).iter()
                    .all(|d| self.completed.contains(d)))
                .collect();

            if ready.is_empty() && self.in_flight.is_empty() {
                break; // All done
            }

            // Launch all ready steps
            for step_ref in &ready {
                match step_ref {
                    StepRef::Compute(partition_id) => {
                        // Fire compute on its stream — non-blocking
                        ctx.launch_compute(*partition_id)?;
                        self.completed.insert(*step_ref);
                    }
                    StepRef::Comm(idx) => {
                        let comm_step = &self.extension.comm_steps[*idx];
                        match &comm_step.op {
                            CollectiveOp::Barrier { group } => {
                                // Barrier returns Result<()>, not CommHandle
                                ctx.barrier(*group).await?;
                                self.completed.insert(*step_ref);
                            }
                            _ => {
                                let handle = ctx.launch_collective(
                                    &comm_step.op,
                                    comm_step.group,
                                    comm_step.stream,
                                )?;
                                self.in_flight.insert(*step_ref, handle);
                            }
                        }
                    }
                }
            }

            // Wait for ANY in-flight operation to complete (not all!)
            if !self.in_flight.is_empty() {
                let completed_ref = self.wait_any().await?;
                self.completed.insert(completed_ref);
                self.in_flight.remove(&completed_ref);
            }
        }
        Ok(())
    }

    async fn wait_any(&self) -> Result<StepRef> {
        loop {
            for (step_ref, handle) in &self.in_flight {
                if handle.is_complete() {
                    return Ok(*step_ref);
                }
            }
            tokio::task::yield_now().await;
        }
    }

    fn all_steps(&self) -> impl Iterator<Item = StepRef> + '_ {
        let compute_refs = self.frozen_plan.partitions.keys()
            .map(|id| StepRef::Compute(*id));
        let comm_refs = (0..self.extension.comm_steps.len())
            .map(StepRef::Comm);
        compute_refs.chain(comm_refs)
    }

    fn deps_of(&self, step: &StepRef) -> &[StepRef] {
        let idx = match step {
            StepRef::Compute(id) => id.0 as usize,
            StepRef::Comm(i) => self.frozen_plan.partitions.len() + i,
        };
        &self.extension.deps[idx]
    }
}
```

### 8.4 Executor Entry Point

```rust
/// Executes a distributed plan across ranks.
///
/// Distributed execution reuses the same `FrozenPlan` / `PartitionTarget` /
/// `PartitionId` model as single-device execution. Each compute step
/// references a compiled partition artifact. Communication steps select
/// their sub-communicator via `GroupId` from the pre-compiled `GroupRegistry`.
pub struct DistributedExecutor {
    frozen_plan: Arc<FrozenPlan>,
    extension: Arc<DistributedPlanExtension>,
    group_registry: GroupRegistry,
}

impl DistributedExecutor {
    /// Execute one forward pass using async DAG scheduling.
    pub async fn forward(
        &self,
        rank: RankId,
        inputs: &[Tensor],
    ) -> Result<Vec<Tensor>> {
        let ctx = ExecutionContext::new(
            &self.frozen_plan,
            &self.group_registry,
            rank,
            inputs,
        )?;

        let mut scheduler = DagScheduler {
            frozen_plan: Arc::clone(&self.frozen_plan),
            extension: Arc::clone(&self.extension),
            in_flight: HashMap::new(),
            completed: HashSet::new(),
        };

        scheduler.execute(&ctx).await?;
        Ok(ctx.collect_outputs(rank))
    }
}
```

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
    ep_group: GroupId,
}

#[async_trait]
impl DispatchTransport for CommunicatorDispatchTransport {
    async fn send(&self, target: GpuId, data: &Tensor) -> Result<()> {
        self.comm.send(&data.buffer, data.len(), data.dtype, RankId(target.0)).await
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
| Data plane (NCCL collectives in GPU graph) | `DistributedPlanExtension` comm steps |
| Dispatch plan table | `ExpertParallel.placement` |
| `MoeDispatchOp` (custom ONNX op) | `PlanStep::Collective(AllToAll)` |
| `ExpertSession` trait | `ExecutionProvider` + rank-specific subgraph |

### 9.3 GPU-Native Mode

For maximum performance (MOE_EXPERT_PARALLELISM.md §5.3 Mode 1), the runtime
**lowers** communication plan ops into rank-local CUDA graph segments. Each rank
launches its own rank-local plan in the validated collective order.

- GPU-native mode = the runtime compiles plan ops into CUDA-graph-capturable
  communication calls that execute alongside EP compute kernels in the same
  captured graph.
- CUDA graph capture captures both compute kernels and communication calls as
  a single graph per rank.
- The EP never creates its own communication — it executes compiled plan
  fragments that include both compute and comm operations.
- Every rank launches independently; collective ordering is guaranteed by
  the plan's `collective_sequences`.
- The communicator completion/error contract is the same as orchestrated mode.

The `ExecutionPlan`/`DistributedExecutor` approach (explicit interleaving) is
the **orchestrated mode** (Mode 2) — more flexible, inspectable, and required
for heterogeneous EP mixing where collectives can't live inside any single EP.

---

## 10. Mac Studio Cluster as First-Class Target

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
layers alongside its expert shard. For released open-weight MoE models
(e.g., Mixtral 8x22B, DeepSeek-V2), the attention weights easily fit within
512 GB alongside the expert shard.

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

### 10.4 Performance Validation

Latency analysis for this cluster target uses released open-weight MoE models
(e.g., Mixtral 8x22B, DeepSeek-V2) as benchmarks. K3-class hypothetical
workloads are deferred to [Appendix A](#appendix-a-hypothetical-workloads-pending-validation).

---

## 11. Phased Implementation

### Phase 1: InProcessCommunicator + Simulated Multi-EP

**Goal:** Validate the abstraction without real hardware.

- Implement `Communicator` trait and `InProcessCommunicator`
- Implement `TensorParallel` and `ExpertParallel` strategy structs
- Implement `DistributedPlanExtension` compilation from strategy + graph
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
- Fault detection: abort all ranks and restart (consistent with MEMORY_ARCHITECTURE.md decision). Partial recovery/degraded execution deferred to Phase 4+.
- **Target:** Released open-weight MoE model (e.g., Mixtral 8x22B, DeepSeek-V2). K3-class workloads deferred to [Appendix A](#appendix-a-hypothetical-workloads-pending-validation) pending model release.

- **Acceptance criteria:** both documents use one failure-state machine and
  define communicator abort, request failure, cleanup, and restart ownership.

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
> Cross-session memory coordination, the `ClusterCoordinator` trait, budget
> arbitration, shared weight deduplication, and the relationship between
> governors and coordinators are consolidated there.
>
> For memory pressure protocol (eviction signaling across ranks), see
> MEMORY_ARCHITECTURE.md §6 (canonical definition).


## 13. Open Questions

> Renumbered from §12. Questions resolved in MEMORY_ARCHITECTURE.md are marked
> as such; only genuinely open questions remain.

1. **Rendezvous mechanism.** **Resolved.** See MEMORY_ARCHITECTURE.md §12, Q1.

2. **Fault tolerance.** **Resolved.** See MEMORY_ARCHITECTURE.md §12, Q2.
   (Abort all ranks and restart; partial recovery deferred to Phase 4+.)

3. **Dynamic rank membership.** **Resolved.** See MEMORY_ARCHITECTURE.md §12, Q3.

4. **Communicator selection.** **Resolved.** See MEMORY_ARCHITECTURE.md §12, Q4.

5. **Async overlap strategy.** How to maximize compute/communication overlap in
   the plan compiler? NCCL supports CUDA stream-based overlap. For TB5/RDMA,
   can we overlap host-side communication with MLX compute? This is critical
   for hiding the TB5 latency.
   *Decision owner: runtime team. Target: Phase 2.*

6. **Collective algorithm selection.** Should the runtime auto-select ring vs
   tree vs direct based on message size and topology, or should the strategy
   layer hint?
   *Decision owner: communicator backend. Target: Phase 2-3.*

7. **Quantized communication.** **Resolved.** See MEMORY_ARCHITECTURE.md §12, Q7.

8. **Speculative dispatch batching.** Can expert dispatch be batched across
   multiple decode steps? Multiple draft tokens route to different experts.
   Batched AllToAll is more efficient but increases latency for the first draft.
   *Decision owner: MoE strategy layer. Target: Phase 3.*

9. **HETEROGENEOUS_PLACEMENT.md integration.** The ON HOLD single-machine
   CPU+CUDA fallback design should eventually compose with distributed placement.
   When a remote node has an unsupported op, does it fall back locally (CPU on
   that node) or route to another node? Likely local fallback first.
   *Decision owner: partitioner. Target: Phase 4.*

10. **Memory pressure across ranks.** If one rank runs out of KV cache memory,
    should it signal other ranks to evict sequences cooperatively? Or does the
    ClusterCoordinator manage this centrally?
    *Decision owner: ClusterCoordinator. Target: Phase 3-4.*

11. **CUDA IPC ownership semantics.** **Resolved.** See MEMORY_ARCHITECTURE.md §12, Q11.

12. **KV cache sharing granularity.** **Resolved.** See MEMORY_ARCHITECTURE.md §12, Q12.

13. **ClusterCoordinator placement.** **Resolved.** See MEMORY_ARCHITECTURE.md §12, Q13.

---

## Appendix A: Hypothetical Workloads (Pending Validation)

> **All figures below are theoretical upper bounds. Do not use for capacity
> planning or milestone acceptance.**

### K3-Class Model (Hypothetical)

Kimi K3 is used as a hypothetical reference workload. The model has not been
publicly released at time of writing. All capacity and throughput figures are
upper-bound estimates derived from published architecture papers (896 experts,
2.8T parameters). This section will be updated with measured benchmarks when
a reproducible model is available.

**Reference configuration:** 4× Mac Studio M3 Ultra (512 GB each) = 2 TB total.
Fits K3-class 2.8T model (FP4) + ~500 GB headroom for KV cache.

**Latency analysis (hypothetical):**

K3-class model (896 experts, top-16, 4096 hidden dim, BF16).
All figures are upper-bound estimates; peak link rate (12 GB/s) is used
rather than measured collective throughput, which will be lower due to protocol
overhead, topology contention, and fixed collective latency:

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

The Mac Studio cluster is **estimated to be compute-bound** given the above
assumptions — TB5 bandwidth appears sufficient because expert dispatch volumes
are modest. This will be validated with measured benchmarks when a reproducible
model is available.

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
