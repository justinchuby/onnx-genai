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
14. [Deferred Workload Validation](#14-deferred-workload-validation)
15. [References](#15-references)

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

    /// This participant's immutable world-rank identity.
    fn rank(&self) -> RankId;

    /// Identity of this communicator's pre-compiled group.
    fn group_id(&self) -> GroupId;

    /// Group members in the frozen peer-vector order.
    fn members(&self) -> &[RankId];

    fn group_size(&self) -> usize { self.members().len() }

    /// Human-readable backend name (e.g., "nccl", "gloo", "thunderbolt").
    fn backend_name(&self) -> &str;

    // ── Collective operations ──
    //
    // Data-transfer and synchronization operations return a `CommHandle`
    // representing an asynchronous, rank-local completion fence.
    // `exchange_counts` completes its small preparatory phase before returning
    // a ticket, and `abort` is a terminal control-plane operation.
    //
    //   - "Enqueued":  the call returns `Ok(CommHandle)` once the operation
    //                  has been submitted to the transport. Device work may
    //                  still be in-flight.
    //   - "Terminal":  awaiting the handle returns `Ok(())` when this rank's
    //                  output is visible and all local input/output buffers are
    //                  safe to reuse. It does NOT prove that every remote rank
    //                  has retired its local operation.
    //
    // **Buffer reuse rule:** the caller MUST NOT reuse, free, or mutate
    // input buffers until the handle reaches a terminal state. The handle
    // retains buffer leases in the communicator's outstanding-operation
    // registry. That registry owns transport progress and leases until terminal,
    // so dropping the Rust handle cannot stop progress or release storage.
    //
    // **Cancellation:** dropping a handle detaches the caller; it NEVER cancels
    // a collective. Cancellation would let ranks diverge in collective order.
    // Abort is an explicit communicator-wide operation owned by the request
    // failure state machine.
    //
    // **Error propagation:** argument/enqueue errors are returned by the method.
    // Asynchronous transport errors are returned by the handle Future. On a
    // collective failure, the executor aborts the communicator before retiring
    // any dependent step.

    /// In-place all-reduce: every rank ends with the element-wise reduction
    /// of all inputs. Default op is sum.
    async fn all_reduce(
        &self,
        instance: CommInstanceId,
        tensor: &mut DeviceBuffer,
        len: usize,
        dtype: DType,
        op: ReduceOp,
    ) -> Result<CommHandle>;

    /// All-to-all: each rank sends a distinct chunk to every other rank and
    /// receives a distinct chunk from every other rank.
    ///
    /// `send_bufs[i]` is sent to `members()[i]`; `recv_bufs[i]` receives from
    /// that world rank. All per-peer vectors use this frozen member order.
    async fn all_to_all(
        &self,
        instance: CommInstanceId,
        send_bufs: &[&DeviceBuffer],
        recv_bufs: &mut [&mut DeviceBuffer],
        chunk_sizes: &[usize],
        dtype: DType,
    ) -> Result<CommHandle>;

    /// Prepare one variable-size all-to-all invocation.
    ///
    /// Counts are logical elements per peer. The returned ticket binds the
    /// exchanged peer counts, wire codec, group, and collective sequence
    /// instance. It is consumed exactly once by `all_to_all_v`.
    async fn exchange_counts(
        &self,
        instance: CommInstanceId,
        send_counts: &[usize],
        spec: &WireTensorSpec,
    ) -> Result<AllToAllVTicket>;

    /// Variable-size all-to-all data transfer.
    ///
    /// `send_offsets` and `recv_offsets` are byte offsets. For every peer, the
    /// implementation computes `offset.checked_add(spec.encoded_bytes(count)?)`
    /// and rejects overflow, out-of-bounds extents, invalid codec alignment,
    /// and overlapping receive spans. The ticket proves that each local receive
    /// count equals the corresponding remote send count. It also validates that
    /// both offset vectors have `group_size()` entries and that ticket group,
    /// instance, and frozen wire spec match this plan operation.
    async fn all_to_all_v(
        &self,
        send_buf: &DeviceBuffer,
        send_offsets: &[usize],
        recv_buf: &mut DeviceBuffer,
        recv_offsets: &[usize],
        ticket: AllToAllVTicket,
    ) -> Result<CommHandle>;

    /// All-gather: each rank contributes a chunk; every rank receives the
    /// concatenation of all chunks.
    async fn all_gather(
        &self,
        instance: CommInstanceId,
        send_buf: &DeviceBuffer,
        recv_buf: &mut DeviceBuffer,
        count: usize,
        dtype: DType,
    ) -> Result<CommHandle>;

    /// Broadcast: rank `root` sends; all other ranks receive.
    async fn broadcast(
        &self,
        instance: CommInstanceId,
        buffer: &mut DeviceBuffer,
        len: usize,
        dtype: DType,
        root: RankId,
    ) -> Result<CommHandle>;

    /// Reduce-scatter: reduce + scatter in one step. Each rank ends with
    /// 1/group_size of the reduced result.
    async fn reduce_scatter(
        &self,
        instance: CommInstanceId,
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
        instance: CommInstanceId,
        buffer: &DeviceBuffer,
        len: usize,
        dtype: DType,
        dest: RankId,
    ) -> Result<CommHandle>;

    /// Receive a buffer from a specific rank.
    async fn recv(
        &self,
        instance: CommInstanceId,
        buffer: &mut DeviceBuffer,
        len: usize,
        dtype: DType,
        source: RankId,
    ) -> Result<CommHandle>;

    // ── Synchronization ──

    /// Asynchronous barrier. Its local handle becomes terminal only after every
    /// rank in this communicator has entered the same barrier instance.
    async fn barrier(&self, instance: CommInstanceId) -> Result<CommHandle>;

    /// Abort all outstanding operations on this communicator. This is idempotent
    /// and transitions every outstanding handle to a terminal error.
    async fn abort(&self, cause: CommError) -> Result<()>;
}

/// Immutable world-rank identity. Group-local vector positions are not ranks.
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

/// Completion handle for one rank's asynchronous operation.
/// Implements `Future<Output = Result<()>>`; polling returns transport errors.
pub struct CommHandle {
    inner: Pin<Box<dyn CommCompletion>>,
}

pub trait CommCompletion: Future<Output = Result<()>> + Send {
    /// Attach as a dependency on a device stream/queue.
    /// The stream will not proceed past this point until the comm is complete.
    fn attach_to_stream(&self, stream: &dyn DeviceStream) -> Result<()>;
}

/// Monotonic execution identity. It is not reused while any operation from the
/// execution is outstanding.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub struct ExecutionId(pub u64);

/// Frozen, group-local submission position assigned by the plan compiler.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub struct CommSequenceId(pub u32);

/// Runtime identity for one communication operation. The pair is unique within
/// a GroupId and also acts as the send/recv message tag.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub struct CommInstanceId {
    pub execution: ExecutionId,
    pub sequence: CommSequenceId,
}

/// Opaque, single-use result of count exchange.
pub struct AllToAllVTicket {
    group: GroupId,
    instance: CommInstanceId,
    send_counts: Vec<usize>,
    recv_counts: Vec<usize>,
    spec: WireTensorSpec,
}

/// Wire codec negotiated and frozen at plan compilation.
pub struct WireTensorSpec {
    pub logical_dtype: DType,
    pub wire_dtype: DType,
    pub codec: WireCodec,
    pub error_bound: f64,
}

impl WireTensorSpec {
    /// Checked encoded byte length, including codec block metadata/padding.
    pub fn encoded_bytes(&self, logical_elements: usize) -> Result<usize>;
    pub fn validate_segment_alignment(&self, byte_offset: usize) -> Result<()>;
}

pub enum WireCodec {
    Identity,
    BlockQuantized {
        block_size: usize,
        scale_dtype: DType,
    },
}
```

`WireTensorSpec` is a transport projection produced by the plan compiler from
the negotiated boundary `TensorFormat`; callers do not construct an unrelated
spec at execution time. Plan validation requires its logical/wire dtypes and
codec to agree with the source converter, destination converter, and reserved
buffer capacities. Reduction collectives use identity/full-precision wire
format in Phases 1-2; codec-aware reduction semantics are a separate Phase 3+
capability and must not be inferred from payload-only quantization support.

`all_to_all_v` preserves order within each peer segment. MoE token-order
reconstruction is not a transport concern: the immutable expert-routing plan
owns the forward permutation and inverse permutation, and the combine step
applies the inverse after the receive completes.

Count exchange does not authorize unbudgeted allocation. The frozen buffer plan
reserves a maximum encoded send/receive workspace for each all-to-all-v step.
`exchange_counts` rejects any peer or total count whose checked encoded extent
exceeds that reservation; only then may the data phase consume the ticket.
Count exchange plus data transfer form one composite `CommInstanceId` for
ordering and failure purposes.

The transport-held lease and detach/abort lifetime are modeled in
[`BufferOwnership.tla`](../specs/tla/BufferOwnership.tla). Any implementation
optimization must preserve its registry-ownership and terminal-release
invariants.

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
    /// Sorted, unique world ranks; this order defines all peer-vector indices.
    pub ranks: Vec<RankId>,
}

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct GroupId(pub u32);
```

Groups are **not** created lazily at runtime. The plan compiler derives all
required groups from the parallel strategy, validates membership, and compiles
them in a single globally-ordered pass. This prevents deadlocks from ranks
creating groups in different orders.

### 3.2.1 Runtime Collective Ordering

Collective transports such as NCCL do not match operations by application tag.
Each `CommGroup` therefore owns a submit sequencer:

- The request coordinator assigns globally monotonic `ExecutionId`s and
  announces admitted executions to every participating rank.
- Within a group, collective enqueue is released in lexicographic
  `(ExecutionId, CommSequenceId)` order. Multiple collectives may remain
  in-flight, but every member submits them in the same order.
- A rank that reaches a later execution early waits in the submit sequencer; it
  does not call the transport out of order.
- Admission failure or cancellation before submission is a coordinator-issued
  skip record observed by all group members. Failure after any member submits
  triggers communicator abort. A rank never silently omits an instance.
- Point-to-point operations use `CommInstanceId` as an actual message tag and
  are validated by the frozen `message_pairs` table.

The sequencer is control-plane state. It never holds device, buffer, or governor
locks while waiting for the next admitted instance.

The overlapping-execution, skip, abort, and rank-local completion contract is
modeled in
[`CollectiveOrdering.tla`](../specs/tla/CollectiveOrdering.tla).

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
    group_id: GroupId,
    members: Arc<[RankId]>,
    /// Backend ordinal derived from `members`, never exposed as RankId.
    transport_ordinal: u32,
    rank: RankId,
}

impl NcclCommunicator {
    /// Initialize from a unique ID shared across all ranks.
    ///
    /// Rank 0 generates the ID; other ranks receive it via a rendezvous
    /// mechanism (e.g., shared file, TCP socket, environment variable).
    pub fn new(
        unique_id: NcclUniqueId,
        group: &GroupSpec,
        rank: RankId,
    ) -> Result<Self>;
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
    group_id: GroupId,
    members: Arc<[RankId]>,
    rank: RankId,
}
```

### 4.4 ThunderboltCommunicator

```rust
/// Thunderbolt 5 communicator for Mac Studio clusters.
///
/// Uses an RDMA-capable transport over TB5 when the deployed stack supports it.
/// Effective bandwidth/latency are measured at startup, not hard-coded.
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
    group_id: GroupId,
    members: Arc<[RankId]>,
    rank: RankId,
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
    group_id: GroupId,
    members: Arc<[RankId]>,
    rank: RankId,
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
    group_id: GroupId,
    members: Arc<[RankId]>,
    rank: RankId,
}

struct InProcessSharedState {
    /// Async barrier keyed by the runtime operation instance.
    barriers: HashMap<CommInstanceId, AsyncBarrier>,
    /// Mailboxes indexed by frozen member-vector positions, not RankId values.
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
    pub node: ClusterNodeId,
    pub local: LocalDeviceId,
}

/// Opaque cluster-machine identifier assigned during rendezvous.
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct ClusterNodeId(pub u32);

/// Local device ordinal within a node (e.g., CUDA:0, CUDA:1).
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct LocalDeviceId {
    pub kind: DeviceKind,
    pub ordinal: u32,
}

impl LocalDeviceId {
    pub const fn cpu() -> Self {
        Self { kind: DeviceKind::Cpu, ordinal: 0 }
    }
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
`GlobalDeviceId` includes the `ClusterNodeId`. The `topo_index` map provides validated
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

These are orientation values, not planner constants or acceptance claims. The
runtime cost model uses topology-specific startup measurements (with an
explicit configured fallback when probing is unavailable) and records the
measurement method alongside the frozen plan.

### 7.3 Placement Constraints

The ILP must respect:

```rust
/// Constraints the graph partitioner respects during distributed placement.
pub struct PlacementConstraints {
    /// Memory budget per device (bytes). Placement must not exceed.
    pub memory_budgets: HashMap<GlobalDeviceId, u64>,
    /// Nodes that MUST be on the same device (e.g., attention Q/K/V).
    pub colocate: Vec<Vec<onnx_runtime_ir::NodeId>>,
    /// Nodes that MUST be on a specific device (e.g., EP claims).
    pub pin: Vec<(onnx_runtime_ir::NodeId, GlobalDeviceId)>,
    /// Maximum allowed communication volume (bytes) across a boundary
    /// class (e.g., "cross-node" < 1 GB per step).
    pub max_cross_boundary_bytes: Option<u64>,
}
```

---

## 8. Distributed Execution Plan

### 8.1 Compilation — One FrozenPlan

Distributed compilation preserves the graph/partition ownership model from
[GRAPHVIEW_LENS_DESIGN.md](./GRAPHVIEW_LENS_DESIGN.md), but does not bolt on a
second dependency graph. **This section is canonical for `FrozenPlan` execution
fields and supersedes the earlier `execution_order: Vec<PlanStep>` sketch in
that document.** The common `PlanStep` enum admits compute and communication
steps, and one dense `StepId` domain indexes the final DAG:

```rust
pub struct FrozenPlan {
    frozen: Arc<FrozenGraph>,
    node_placement: Vec<Option<PartitionTarget>>,
    value_placement: Vec<Option<PartitionTarget>>,
    partitions: Vec<PartitionDescriptor>,
    /// The sole authoritative execution DAG. `StepId` is the dense index.
    steps: Vec<PlanStep>,
    distributed: Option<DistributedMetadata>,
}

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub struct StepId(pub u32);

pub struct PlanStep {
    pub id: StepId,
    /// Exactly the ranks that execute this step.
    pub participants: RankSet,
    pub deps: Vec<StepId>,
    pub kind: PlanStepKind,
}

pub enum PlanStepKind {
    /// `PartitionId` is opaque and is resolved through `FrozenPlan::partition`.
    Compute { partition: PartitionId, stream: StreamId },
    Communication(CommunicationStep),
}

pub struct CommunicationStep {
    pub op: CommunicationOp,
    pub group: GroupId,
    /// Frozen sequence; executor combines it with the current ExecutionId.
    pub sequence: CommSequenceId,
    pub stream: StreamId,
    pub reads: Vec<BufferId>,
    pub writes: Vec<BufferId>,
}

pub enum CommunicationOp {
    Collective(CollectiveOp),
    Send { destination: RankId },
    Recv { source: RankId },
    Barrier,
}

pub struct DistributedMetadata {
    pub groups: Vec<GroupSpec>,
    /// Expected collective submission order for every full group.
    pub collective_sequences: HashMap<GroupId, Vec<StepId>>,
    /// Exactly one send step and one recv step for each point-to-point tag.
    pub message_pairs: HashMap<(GroupId, CommSequenceId), (StepId, StepId)>,
}

pub struct RankSet(BitVec);

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub struct StreamId(pub u32);

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub struct BufferId(pub u32);
```

The plan compiler assigns dense `StepId`s only after partition and communication
insertion. `PartitionId`, graph `NodeId`, and `StepId` are separate opaque
identity domains and are never cast into one another.

Before freeze, release-mode validation proves:

- Every `PlanStep.id` equals its dense index and every dependency exists.
- The DAG is acyclic; a pending step with no possible predecessor completion is
  an invalid plan, not successful execution.
- For every participating rank, every dependency also participates on that rank.
  Cross-rank dependencies must be represented by a communication step.
- A compute step participates only on ranks selected by its
  `PartitionTarget`; non-members never dispatch the partition.
- For each collective `GroupId`, every member rank submits the same
  `CommSequenceId` sequence. At execution, every rank combines it with the same
  `ExecutionId`; non-members submit none of those collectives.
- Every point-to-point `(GroupId, CommSequenceId)` has exactly one send and one
  receive with matching source, destination, byte extent, and wire format.
- Every read/write buffer has a plan-owned lease covering the step's terminal
  completion, and output collection references only rank-local terminal steps.

This single DAG enables:

- **Overlap:** Independent compute partitions on different streams execute
  concurrently with communication on the comm stream.
- **Correctness:** one validated rank-local view and one collective sequence
  derive from the same frozen steps.
- **Buffer safety:** buffer leases retire only when all reader/writer steps are
  terminal.
- **No duplicate authority:** placement, dependencies, communication, and
  execution order are all fields of the same frozen plan.

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

The executor derives an immutable rank-local view from `FrozenPlan`, launches
each ready `Pending` step exactly once, and waits for any compute or
communication completion. Enqueue is not completion: compute kernels return a
device event just as communication returns a `CommHandle`.

For a communication step, `ExecutionContext::enqueue` constructs
`CommInstanceId { execution: ctx.execution_id, sequence: step.sequence }`.
Collectives use it for ordering; point-to-point send/recv use the same pair as
their message tag.

```rust
enum StepState {
    Pending,
    InFlight,
    Terminal,
}

/// Both variants provide the terminal fence after transport/device enqueue.
enum StepCompletion {
    Compute(DeviceEvent),
    Communication(CommHandle),
}

pub struct DagScheduler<'plan> {
    plan: &'plan FrozenPlan,
    rank: RankId,
    local_steps: Vec<StepId>, // sorted by StepId
    states: Vec<StepState>,   // StepId-indexed
}

impl DagScheduler<'_> {
    pub async fn execute(&mut self, ctx: &ExecutionContext) -> Result<()> {
        let mut in_flight = FuturesUnordered::new();
        let mut terminal_count = 0usize;

        loop {
            let ready: Vec<StepId> = self.local_steps.iter().copied()
                .filter(|id| matches!(self.states[id.0 as usize], StepState::Pending))
                .filter(|id| self.plan.step(*id).deps.iter()
                    .all(|dep| matches!(
                        self.states[dep.0 as usize],
                        StepState::Terminal
                    )))
                .collect();

            for id in ready {
                // Transition before polling so this step cannot be relaunched.
                self.states[id.0 as usize] = StepState::InFlight;
                let step = self.plan.step(id);
                in_flight.push(async move {
                    // Includes submit-sequencer wait, enqueue, and terminal
                    // device/transport completion. Waiting here does not block
                    // polling of any other ready step.
                    (id, ctx.run_to_terminal(step).await)
                });
            }

            if terminal_count == self.local_steps.len() {
                return Ok(());
            }

            match in_flight.next().await {
                Some((id, Ok(()))) => {
                    self.states[id.0 as usize] = StepState::Terminal;
                    terminal_count += 1;
                }
                Some((_id, Err(error))) => {
                    // Abort preserves collective ordering: no rank continues
                    // submitting after one rank observes a terminal failure.
                    let abort_error = ctx.abort_communicators(&error).await.err();
                    // Device work may be non-cancellable. Wait for every local
                    // step registry entry to become terminal before releasing
                    // its leases or returning the failed request.
                    let quiesce_error = ctx.quiesce_outstanding_steps().await.err();
                    return Err(error.with_cleanup_context(
                        abort_error,
                        quiesce_error,
                    ));
                }
                None => {
                    return Err(Error::InvalidPlan(
                        "rank-local DAG is cyclic or has an unsatisfied dependency"
                    ));
                }
            }
        }
    }
}
```

`FuturesUnordered` polls terminal results, including transport errors; the
scheduler never busy-spins on a boolean. `ExecutionContext` also registers every
submitted compute and communication step independently of these Rust Futures,
so error cleanup can quiesce work even after the scheduler drops its local
future set. A future optimization may attach a dependency fence directly to a
consumer stream and submit that consumer before host observation, but it must
preserve the same state transition and buffer lease rules.

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
    group_registry: GroupRegistry,
}

impl DistributedExecutor {
    /// Execute one forward pass using async DAG scheduling.
    pub async fn forward(
        &self,
        execution: ExecutionId,
        rank: RankId,
        inputs: &[Tensor],
    ) -> Result<Vec<Tensor>> {
        let ctx = ExecutionContext::new(
            &self.frozen_plan,
            &self.group_registry,
            execution,
            rank,
            inputs,
        )?;

        let mut scheduler = DagScheduler::for_rank(&self.frozen_plan, rank)?;

        scheduler.execute(&ctx).await?;
        Ok(ctx.collect_outputs(rank))
    }
}
```

The request coordinator allocates one `ExecutionId` and distributes it to all
participating ranks before any rank submits the first step. A rank never derives
it from a local counter: retries, admission failure, and overlapping requests
would otherwise produce different message tags. The ID remains live until all
rank-local operations are terminal or communicator abort has drained them.

### 8.5 Failure State Machine

The request coordinator is the sole owner of distributed execution failure:

```text
Running
  ├─ local enqueue/completion error ─► Aborting
  ├─ rank heartbeat/transport loss ──► Aborting
  └─ success on every rank ──────────► Complete

Aborting ─► Quiescing ─► Failed ─► Restarting(new topology epoch)
```

1. The first observer reports `(ExecutionId, rank, cause, topology_epoch)`.
   The coordinator atomically transitions `Running -> Aborting`; later reports
   attach diagnostics but do not create another recovery owner.
2. Every reachable rank stops new submission for that execution and its later
   queued executions, aborts affected communicators, and quiesces local compute,
   transport progress, and allocation leases.
3. Each reachable rank acknowledges `Quiesced`. A crashed rank cannot
   acknowledge; transport timeout destroys the old communicator/process group
   and fences its topology epoch.
4. The coordinator fails the user request exactly once. No output or cache
   mutation from the failed execution is committed after `Aborting`.
5. Phase 1-3 recovery recreates the full rank group with a new topology epoch
   and new `ExecutionId`s. It may reuse coordinator-owned immutable host weight
   mappings, but never device state or allocations owned by the failed group.
   Later queued executions from the fenced epoch are re-admitted with new IDs
   or failed; they are never resumed under their old communication instances.

Partial rank replacement and degraded execution are Phase 4+ work. Until then,
no backend may continue a surviving subset after collective failure.

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

**Migration path:** new MoE plans emit `PlanStepKind::Communication` directly.
`DispatchTransport` is not a lossless abstraction: its `recv` allocates an
unspecified tensor and its methods hide completion fences and wire format.
During migration only, an adapter may wrap an already-resolved expert subgroup
plus an allocator/format policy. It must await the `CommHandle` before returning
the legacy `Result<()>`:

```rust
pub struct CommunicatorDispatchTransport {
    /// Resolved from GroupRegistry at construction; this is already the EP group.
    subgroup: Arc<dyn Communicator>,
    allocator: Arc<dyn ReceiveAllocator>,
    wire_spec: WireTensorSpec,
    gpu_to_rank: HashMap<GpuId, RankId>,
    ids: Arc<dyn LegacyCommIdSource>,
}

#[async_trait]
impl DispatchTransport for CommunicatorDispatchTransport {
    async fn send(&self, target: GpuId, data: &Tensor) -> Result<()> {
        let rank = *self.gpu_to_rank.get(&target)
            .ok_or(Error::UnknownLegacyGpu(target))?;
        let instance = self.ids.next_send(target)?;
        let completion = self.subgroup
            .send(instance, &data.buffer, data.len(), data.dtype, rank)
            .await?;
        completion.await
    }

    // recv/all_to_all perform explicit allocation and format validation,
    // then await their completion handle before satisfying the legacy API.
}
```

The adapter is deleted once the old MoE call sites move to frozen plan steps;
it is not a second production communication contract. `LegacyCommIdSource`
receives coordinator-issued execution IDs and a frozen dispatch sequence; it
must not generate tags from an unsynchronized local counter.

### 9.2 Control Plane / Data Plane Preservation

The control plane / data plane separation from MOE_EXPERT_PARALLELISM.md §2
maps directly to this design:

| MoE Doc Concept | Distributed Runtime Equivalent |
|---|---|
| Control plane (expert placement, rebalancing) | `HybridStrategy` + `ExpertPlacement` |
| Data plane (NCCL collectives in GPU graph) | `FrozenPlan` communication steps |
| Dispatch plan table | `ExpertParallel.placement` |
| `MoeDispatchOp` (custom ONNX op) | `PlanStepKind::Communication` |
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

The `FrozenPlan`/`DistributedExecutor` approach (explicit interleaving) is
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
workloads are deferred to [§14](#14-deferred-workload-validation).

---

## 11. Phased Implementation

### Phase 1: InProcessCommunicator + Simulated Multi-EP

**Goal:** Validate the abstraction without real hardware.

- Implement `Communicator` trait and `InProcessCommunicator`
- Implement `TensorParallel` and `ExpertParallel` strategy structs
- Compile compute and communication into one validated `FrozenPlan`
- Test with multiple CPU EPs in one process simulating multi-device
- Verify correctness: distributed execution matches single-device results
- Verify each rank executes only its `participants` steps and non-members never
  submit a subgroup collective.
- Verify an in-flight step is enqueued exactly once, downstream steps wait for
  terminal compute/communication fences, and invalid/cyclic rank-local DAGs fail.
- Inject enqueue and asynchronous completion errors; assert communicator abort,
  local-step quiescence, lease release, and no subsequent collective submission.
- Run overlapping executions of the same frozen plan and assert
  `CommInstanceId` prevents cross-request send/recv or collective matching.
- Property-test `all_to_all_v` checked extents, offset overflow, overlapping
  receive ranges, mismatched peer counts, ticket reuse, and token permutation
  round-trips.
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
- **Target:** Released open-weight MoE model (e.g., Mixtral 8x22B, DeepSeek-V2). K3-class workloads are deferred to [§14](#14-deferred-workload-validation) pending reproducible artifacts.

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

## 14. Deferred Workload Validation

### 14.1 K3-Class Workload

K3-class capacity and throughput analysis is deferred. This document makes no
fit, bandwidth-sufficiency, latency, bottleneck, or tokens-per-second claim for
an unreleased or non-reproducible workload.

The evaluation may be reopened only after all of the following are available:

- released model artifacts and authoritative architecture/packing metadata;
- measured resident bytes including codec metadata, runtime workspace, KV
  cache, allocator fragmentation, and duplicated/non-sharded regions;
- measured payload throughput and latency for the selected TB5 topology and
  collective implementation, including concurrent-link contention;
- achieved kernel throughput for the released operators and dtypes, not peak
  device FLOPS;
- end-to-end prefill and decode measurements across representative batch,
  context, routing-skew, and failure/restart scenarios.

Any future result belongs in a dated benchmark report. It does not become a
phase acceptance target merely by being copied into this appendix.

---

## 15. References

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
