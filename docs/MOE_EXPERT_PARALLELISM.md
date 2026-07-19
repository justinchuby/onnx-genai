# MoE Expert Parallelism: Session-Per-GPU Architecture

> Companion to [SCHEDULING.md](./SCHEDULING.md) and [DESIGN.md](./DESIGN.md) §14.3 (Multi-GPU).
> Covers expert-parallel deployment of large Mixture-of-Experts models using onnx-genai
> as the orchestrator and ORT sessions as per-GPU execution units.

**Status:** Design Proposal
**Author:** Claw (with Justin)
**Date:** 2026-07-17
**Motivation:** Kimi K3 (2.8T, 896 experts, 16 active) and similar frontier MoE models

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Architecture: Session-Per-GPU with External Dispatch](#2-architecture-session-per-gpu-with-external-dispatch)
3. [Why Not Tensor Parallelism](#3-why-not-tensor-parallelism)
4. [Expert Placement Strategy](#4-expert-placement-strategy)
5. [Token Dispatch Pipeline](#5-token-dispatch-pipeline)
6. [Attention Layer Handling](#6-attention-layer-handling)
7. [KV Cache Implications](#7-kv-cache-implications)
8. [Communication Primitives](#8-communication-primitives)
9. [Integration with nxrt EP Negotiation](#9-integration-with-nxrt-ep-negotiation)
10. [Mac Studio Cluster Considerations](#10-mac-studio-cluster-considerations)
11. [Inference Metadata Extensions](#11-inference-metadata-extensions)
12. [Phased Implementation](#12-phased-implementation)
13. [Open Questions](#13-open-questions)
14. [References](#14-references)

---

## 1. Problem Statement

Frontier MoE models break assumptions baked into current onnx-genai:

| Property | Dense 70B | MoE 2.8T (K3-class) |
|---|---|---|
| Total params | 70B | 2,800B |
| Active params/token | 70B | ~50B (16/896 experts) |
| Weight memory (FP4) | ~35 GB | ~1,400 GB |
| Single GPU? | ✅ (H200 141GB) | ❌ (need 16+ GPUs) |
| Parallelism needed | None or TP | Expert Parallel (EP) |
| Compute bottleneck | Matmul | Router dispatch + all-to-all communication |

**Key insight:** Per-token compute is modest (~50B active params), but the weights are
spread across many devices. The challenge is *routing tokens to the right experts
on the right GPUs* with minimal communication overhead.

Current onnx-genai treats each ORT session as a self-contained execution unit on one
device. This document proposes leveraging that constraint rather than fighting it:
**each GPU runs an ORT session holding a shard of experts, and genai-server orchestrates
the token dispatch externally.**

---

## 2. Architecture: Session-Per-GPU with External Dispatch

```
┌─────────────────────────────────────────────────────────────────┐
│                     genai-server (Rust, async)                   │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │  Router      │  │  Dispatcher  │  │  Combiner             │  │
│  │  (softmax    │  │  (token →    │  │  (weighted sum of     │  │
│  │   top-k)     │  │   GPU map)   │  │   expert outputs)     │  │
│  └──────┬───────┘  └──────┬───────┘  └───────────┬───────────┘  │
│         │                 │                      │               │
├─────────┼─────────────────┼──────────────────────┼───────────────┤
│         ▼                 ▼                      ▼               │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              Expert Dispatch Bus                            │ │
│  │    (CUDA IPC / DLPack zero-copy / TB5 / RDMA)              │ │
│  └─────────┬──────────┬──────────┬──────────┬─────────────────┘ │
│            │          │          │          │                    │
│       ┌────▼────┐┌────▼────┐┌────▼────┐┌────▼────┐             │
│       │ GPU 0   ││ GPU 1   ││ GPU 2   ││ GPU N   │             │
│       │ ORT Ses.││ ORT Ses.││ ORT Ses.││ ORT Ses.│             │
│       │         ││         ││         ││         │             │
│       │ Experts ││ Experts ││ Experts ││ Experts │             │
│       │ 0..55   ││ 56..111 ││ 112..167││ ...     │             │
│       │         ││         ││         ││         │             │
│       │ Attn    ││ Attn    ││ Attn    ││ Attn    │             │
│       │ Shard 0 ││ Shard 1 ││ Shard 2 ││ Shard N │             │
│       └─────────┘└─────────┘└─────────┘└─────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

### Design Principles: Control Plane / Data Plane Separation

The core insight: **separate the "what" (strategy/policy) from the "how" (execution).**

```
┌─────────────────────────────────────────────────────────────────┐
│  CONTROL PLANE (Rust, async) — slow path, flexible, maintainable │
│                                                                  │
│  Responsibilities:                                               │
│  • Expert placement decisions (which expert → which GPU)         │
│  • Compile dispatch plan (lookup table: expert_id → gpu + offset)│
│  • Monitor activation frequencies                                │
│  • Dynamic rebalancing (replicate hot experts, hibernate cold)   │
│  • KV cache eviction policy                                      │
│  • SLA enforcement, session lifecycle                            │
│                                                                  │
│  When it intervenes:                                             │
│  • Startup / model load                                          │
│  • Rebalance events (every N seconds, not every token)           │
│  • Error recovery                                                │
│                                                                  │
│  NOT on the per-token hot path.                                  │
├─────────────────────────────────────────────────────────────────┤
│  DATA PLANE (GPU-native) — hot path, maximum performance         │
│                                                                  │
│  Responsibilities:                                               │
│  • Router forward (on-GPU, produces expert_ids + gate_weights)   │
│  • Token dispatch via NCCL All-to-All (GPU↔GPU, no host)         │
│  • Local expert compute (each GPU runs its shard)                │
│  • Result gather via NCCL All-to-All                             │
│  • Weighted combine (on-GPU)                                     │
│  • Attention forward with TP AllReduce (on-GPU)                  │
│                                                                  │
│  The entire decode loop runs without returning to Rust.           │
│  genai-server "fires" the loop and collects output tokens.       │
└─────────────────────────────────────────────────────────────────┘
```

**Analogy:** Linux networking. Control plane (routing table, iptables) is set in
userspace. Data plane (packet forwarding) runs in kernel fast path — packets never
go back to userspace. Same idea: Rust sets the dispatch plan, GPUs execute it
at wire speed.

### Why This Architecture

1. **Performance = GPU-native.** The decode loop is pure GPU execution with NCCL
   collectives. Zero Rust overhead per token. Equivalent to hand-written multi-GPU
   pipeline (vLLM, Megatron, etc.).

2. **Flexibility = Rust control plane.** Placement strategy, rebalancing policy,
   monitoring, and lifecycle are all in maintainable Rust code. Changing routing
   strategy doesn't require touching CUDA kernels.

3. **Clarity = clean separation.** Data plane is deterministic execution of a compiled
   plan. Control plane is policy/strategy. They interact through a narrow interface
   (the dispatch plan table).

4. **Testable.** Control plane tested with mock GPU metrics. Data plane tested with
   real multi-GPU but deterministic inputs. The plan table is the contract between them.

### How GPU Sessions Execute Without Host Round-Trips

The key mechanism: **the dispatch plan is baked into each ORT session as a constant
tensor.** Each GPU's session graph contains:

```
Router → LookupDispatchPlan → NCCL_AllToAll → LocalExperts → NCCL_AllToAll → Combine
```

This is achieved via a custom ONNX op (`MoeDispatch`) that encapsulates the NCCL call:

```rust
/// Custom ORT op that performs NCCL All-to-All for MoE token dispatch.
/// Registered as a custom op domain in ORT; appears as a single node in the ONNX graph.
///
/// Inputs:
///   - hidden_states: [local_batch, hidden_dim]
///   - expert_ids: [local_batch, top_k] (from router)
///   - dispatch_plan: [num_experts] → int (expert_id → target_rank, constant)
///
/// Outputs:
///   - dispatched_states: [received_tokens, hidden_dim] (tokens routed TO this GPU)
///   - token_metadata: routing info for the gather step
pub struct MoeDispatchOp {
    nccl_comm: NcclCommunicator,
    rank: usize,
    world_size: usize,
}
```

When genai-server wants to change placement (rebalance), it:
1. Pauses the decode loop (after current token completes)
2. Updates the dispatch_plan constant tensor in each session
3. Resumes the loop

This is the only point where Rust touches the hot path — and it's rare (every N seconds
or on-demand, not per-token).

---

## 3. Why Not Tensor Parallelism

TP shards every layer across all GPUs. For MoE this is wasteful:

| Approach | Communication Pattern | Active GPU Utilization |
|---|---|---|
| Tensor Parallel | AllReduce every layer | All GPUs compute every token |
| Expert Parallel | All-to-All per MoE layer | Only GPUs with active experts compute |

For 896-expert models with top-16 routing:
- TP across 16 GPUs: every GPU runs 1/16 of every expert = all GPUs active every token
- EP across 16 GPUs: ~56 experts/GPU, only GPUs with active experts run = ~16/56 ≈ 29% of each GPU's experts activate per token

**EP wastes less compute at the cost of more selective communication.** For very sparse
MoE (top-16 out of 896 = 1.8% density), EP is clearly the right choice.

The attention layers still use TP within the EP framework (see §6).

---

## 4. Expert Placement Strategy

### 4.1 Static Placement

Simplest approach: distribute experts round-robin or contiguously across GPUs.

```rust
/// Static expert-to-GPU assignment.
pub struct ExpertPlacement {
    /// Total number of experts in the model
    pub num_experts: usize,
    /// Number of active experts per token (top-K)
    pub top_k: usize,
    /// Assignment: expert_id → gpu_id
    pub assignment: Vec<GpuId>,
}

impl ExpertPlacement {
    /// Contiguous blocks: GPU 0 gets experts 0..N/G, GPU 1 gets N/G..2N/G, etc.
    pub fn contiguous(num_experts: usize, num_gpus: usize) -> Self;

    /// Round-robin: expert i goes to GPU i % num_gpus.
    /// Better load balance when co-activation patterns are sequential.
    pub fn round_robin(num_experts: usize, num_gpus: usize) -> Self;
}
```

### 4.2 Affinity-Aware Placement

Analyze routing statistics from calibration data to place frequently co-activated
experts on the same GPU, minimizing cross-GPU traffic:

```rust
/// Co-activation affinity matrix from calibration.
pub struct ExpertAffinity {
    /// affinity[i][j] = frequency that expert i and j are both in top-K for same token
    pub coactivation: Vec<Vec<f32>>,
}

impl ExpertAffinity {
    /// Compute from a calibration dataset by running the router on sample inputs.
    pub fn from_calibration(
        router: &OrtSession,
        calibration_data: &[Tensor],
        top_k: usize,
    ) -> Self;

    /// Solve placement as a graph partitioning problem:
    /// maximize intra-GPU co-activation, minimize cross-GPU traffic.
    /// Uses METIS-style multilevel partitioning or simpler greedy clustering.
    pub fn optimal_placement(&self, num_gpus: usize) -> ExpertPlacement;
}
```

### 4.3 Dynamic Expert Replication

For hot experts (disproportionately activated), replicate them across multiple GPUs:

```rust
pub struct DynamicPlacement {
    base: ExpertPlacement,
    /// Experts replicated to additional GPUs. Key = expert_id, Value = extra GPU ids.
    replicas: HashMap<ExpertId, Vec<GpuId>>,
    /// Activation frequency tracker (exponential moving average).
    frequency: Vec<f32>,
}

impl DynamicPlacement {
    /// Periodically rebalance: replicate experts with frequency > threshold,
    /// remove replicas for experts below threshold.
    pub fn rebalance(&mut self, threshold: f32, max_replicas_per_expert: usize);
}
```

---

## 5. Token Dispatch Pipeline

### 5.1 Per-MoE-Layer Execution Flow

```
For each MoE layer:

1. ROUTER FORWARD
   Input: hidden_states [batch, hidden_dim]
   Output: expert_ids [batch, top_k], gate_weights [batch, top_k]
   Runs on: orchestrator GPU or CPU

2. TOKEN GROUPING
   Group tokens by target GPU based on expert_ids + placement map.
   Result: per_gpu_tokens: HashMap<GpuId, (token_indices, expert_ids, hidden_states)>

3. ASYNC DISPATCH
   For each GPU in parallel:
     - Send grouped tokens + target expert_ids to the GPU's ORT session
     - ORT session runs local experts, returns outputs
   Concurrency: tokio::join! over all GPU futures

4. GATHER + COMBINE
   Collect all expert outputs, reorder by original token index.
   Weighted sum: output[i] = Σ_k gate_weights[i,k] * expert_output[i,k]
   Runs on: orchestrator GPU (or whichever GPU feeds into next attention layer)
```

### 5.2 Dispatch Granularity: Full-Graph GPU Execution (Recommended)

With the control plane / data plane separation, the question is not "per-layer vs
per-block" but rather: **the entire model graph runs on GPUs without returning to host.**

Each GPU's ORT session contains the full transformer stack for its shard:
- All attention layers (TP sharded)
- All MoE layers (EP sharded, with NCCL dispatch ops)
- Router networks (replicated on all GPUs)

```
GPU 0 session graph (simplified, one block):
  LayerNorm → Attention(heads 0..H/N) → TP_AllReduce → LayerNorm
  → Router → MoeDispatch(NCCL) → LocalExperts(0..55) → MoeGather(NCCL) → Combine
  → Residual → [next block...]
```

The MoeDispatch and MoeGather custom ops handle cross-GPU token movement internally
via NCCL. From ORT's perspective, they're just ops with tensor inputs/outputs.

**Fallback mode for debugging/testing:** A pure-Rust per-layer dispatch mode
(genai-server orchestrates each layer individually) is useful for:
- Development without multi-GPU
- Unit testing routing logic
- Profiling to identify bottlenecks

This fallback uses the `ExpertSession` trait from §10 with `HostStagedTransport`.

### 5.3 Execution Modes

#### Mode 1: GPU-Native (Production — Maximum Performance)

The full model graph runs on GPUs. genai-server only handles:
- Session setup (load model, set dispatch plan)
- Feeding input tokens → collecting output logits
- Control plane decisions (rebalance, KV eviction)

```rust
/// Production execution: fire the multi-GPU graph and collect output.
pub async fn generate_token(
    sessions: &[GpuSession],  // One per GPU, holding the full model shard
    input_ids: &Tensor,
    kv_caches: &mut [KvCacheHandle],
) -> Result<Tensor> {
    // Each GPU session runs the FULL forward pass for its shard.
    // NCCL ops inside the graph handle cross-GPU communication.
    // We only need to feed input to rank 0 and read logits from rank 0.
    let logits = sessions[0].run_forward(input_ids, &kv_caches[0]).await?;
    Ok(logits)
}
```

#### Mode 2: Orchestrated Dispatch (Development/Debug — Maximum Flexibility)

genai-server orchestrates each layer individually. Slower but inspectable:

```rust
/// Debug/development mode: Rust orchestrates each MoE layer.
/// Allows inspection of routing decisions, activation stats, etc.
pub async fn moe_dispatch_debug(
    hidden_states: &Tensor,
    router_session: &OrtSession,
    expert_sessions: &[GpuExpertSession],
    placement: &ExpertPlacement,
    top_k: usize,
    stats: &mut DispatchStats,  // For monitoring/profiling
) -> Result<Tensor> {
    // 1. Route
    let (expert_ids, gate_weights) = router_session.run_router(hidden_states)?;
    stats.record_routing(&expert_ids);

    // 2. Group tokens by GPU
    let gpu_groups = group_tokens_by_gpu(&expert_ids, placement);

    // 3. Async dispatch to all GPUs
    let expert_outputs: Vec<(GpuId, Tensor)> = futures::future::join_all(
        gpu_groups.into_iter().map(|(gpu_id, group)| {
            let session = &expert_sessions[gpu_id];
            async move {
                let output = session.run_experts(
                    &group.hidden_states,
                    &group.local_expert_ids,
                ).await?;
                Ok::<_, Error>((gpu_id, output))
            }
        })
    ).await.into_iter().collect::<Result<Vec<_>>>()?;

    // 4. Combine
    let output = combine_expert_outputs(
        &expert_outputs, &expert_ids, &gate_weights, placement,
        hidden_states.shape(),
    )?;

    Ok(output)
}
```

#### Switching Between Modes

The mode is a deployment-time choice, not a code path fork:
- **GPU-Native:** Use ONNX graphs that include MoeDispatch/MoeGather custom ops + NCCL
- **Orchestrated:** Use ONNX graphs with only local experts (no NCCL ops); genai-server
  handles dispatch externally

Same model weights, different graph compilation. Mobius (or equivalent exporter) can
emit either variant based on a deployment target flag.

---

## 6. Attention Layer Handling

MoE models typically interleave attention and MoE-FFN layers:

```
Layer N:   [Attention (TP)]  →  [MoE-FFN (EP)]
Layer N+1: [Attention (TP)]  →  [MoE-FFN (EP)]
```

Attention layers are dense (not sparse) — every token uses the full attention weights.
These use **tensor parallelism (TP)** across GPUs, with AllReduce to synchronize.

### Hybrid TP+EP within one session

Each GPU's ORT session contains:
- **Attention shard** (1/N of attention heads, via TP)
- **Expert shard** (M/896 experts, via EP)
- **Shared experts** (if the architecture has them, e.g. DeepSeek V3 has 1 shared expert
  that runs on every GPU)

```
GPU 0 ORT Session ONNX graph:
  ├── attention_layer_0 (heads 0..H/N, with TP AllReduce)
  ├── moe_ffn_layer_0 (experts 0..55, dispatched by orchestrator)
  ├── attention_layer_1 ...
  ├── moe_ffn_layer_1 ...
  └── ...
```

### KDA / AttnRes / Gated MLA

Novel attention mechanisms (Kimi K3's KDA, DeepSeek's MLA, etc.) require:

1. **Custom attention kernels in ORT.** These run inside the ORT session. From genai-server's
   perspective, attention is opaque — we only care about KV cache I/O at the boundaries.

2. **Modified prefix caching logic.** KDA's prefix cache is different from standard MHA.
   The KV cache manager needs per-attention-type caching strategies, keyed by the
   model's `inference_metadata.yaml`.

3. **AttnRes (Attention Residuals):** Selectively retrieves representations from earlier
   layers. This means KV "cache" isn't just per-layer — there may be cross-layer
   references. The page table needs to support cross-layer page sharing if AttnRes
   is declared in metadata.

---

## 7. KV Cache Implications

### 7.1 Distributed KV Cache

With TP attention, each GPU holds KV cache for its shard of attention heads:

```
GPU 0: KV pages for heads 0..H/N    (all sequences)
GPU 1: KV pages for heads H/N..2H/N (all sequences)
...
```

The existing `PageTable` (DESIGN.md §3.2) needs a `device` field on each page,
which it already has (`Page.device: Device`). Extension needed:

```rust
/// Distributed page table: one logical sequence maps to pages across multiple devices.
pub struct DistributedPageTable {
    /// Per-device page tables (each manages its own pages).
    shards: Vec<PageTable>,
    /// Logical sequence → list of (device, page_ids) for each position.
    /// All shards store the same token positions, but different head dimensions.
    sequence_map: HashMap<SequenceId, Vec<DistributedPageRef>>,
}

pub struct DistributedPageRef {
    pub position_range: Range<usize>,
    /// One page per device shard (same position range, different heads).
    pub shard_pages: Vec<(DeviceId, PageId)>,
}
```

### 7.2 KV Cache Memory Budget

For a K3-class model on 16 GPUs (each H200 141 GB):
- Weights per GPU: ~1400 GB / 16 = ~87.5 GB
- Available for KV: 141 - 87.5 = ~53.5 GB per GPU
- KV per token (assuming MLA-style compressed KV): varies by architecture
- With standard GQA at 128 heads / 16 GPUs = 8 heads/GPU:
  8 heads × 2 (K+V) × 128 dim × 2 bytes (bf16) × num_layers ≈ model-specific

Budget computation must account for both the attention shard KV and any
expert-related buffers (input staging, output staging for dispatch).

---

## 8. Communication Primitives

> **DEPRECATED:** The `DispatchTransport` trait defined in this section is superseded
> by the `Communicator` trait. See [MEMORY_ARCHITECTURE.md §6](./MEMORY_ARCHITECTURE.md).
> Use `Communicator` for all new work.

### 8.1 Intra-Node Bandwidth Reference

| Method | Bandwidth | Latency | Use Case |
|---|---|---|---|
| NVLink (H100/H200) | 900 GB/s | <1 μs | AllReduce (attention TP), All-to-All (MoE EP) |
| CUDA IPC (shared mem) | PCIe-limited ~64 GB/s | ~5 μs | Dispatch tokens between sessions without host copy |
| DLPack zero-copy | Same as underlying | Minimal | Share tensors between ORT sessions in genai-server |
| Host staging (PCIe) | ~32 GB/s per direction | ~10 μs | Fallback, always works |

### 8.2 Cross-Node Bandwidth Reference

| Method | Bandwidth | Notes |
|---|---|---|
| TB5 (bilateral) | 80 Gbps (~10 GB/s) | Mac Studio to Mac Studio |
| RDMA / InfiniBand | 200-400 Gbps | Data center multi-server |
| TCP/IP (gRPC) | Limited by NIC | Fallback, high latency |


## 9. Integration with nxrt EP Negotiation

The nxrt EP negotiation protocol (SCHEDULING.md §8) naturally extends to expert parallelism:

### 9.1 EP Capability Declaration

Each "EP" (in nxrt terms) represents a GPU's ORT session. It declares:

```rust
/// Extended EP capabilities for MoE expert parallelism.
pub struct MoeEpCapabilities {
    /// Which expert indices this EP holds.
    pub expert_range: Range<usize>,

    /// Whether this EP also holds shared/always-active experts.
    pub has_shared_experts: bool,

    /// Attention head range (for TP shard).
    pub attention_head_range: Range<usize>,

    /// Available memory for KV cache (after weights loaded).
    pub kv_memory_budget_bytes: usize,

    /// Communication capabilities.
    pub transport: TransportCapabilities,
}

pub struct TransportCapabilities {
    /// Can do CUDA IPC (same-node multi-GPU)?
    pub cuda_ipc: bool,
    /// Available NVLink bandwidth (0 if none).
    pub nvlink_bandwidth_gbps: f32,
    /// Network bandwidth to other nodes.
    pub network_bandwidth_gbps: f32,
}
```

### 9.2 Dynamic Expert Scheduling (Justin's Proposal)

From Justin's nxrt design notes: "根据外部资源信息runtime动态调控EP使用."

For MoE, this means the scheduler can:

1. **Monitor expert activation frequencies** in real time.
2. **Migrate hot experts** to GPUs with more headroom.
3. **Shed load** by routing tokens to replicated experts on less-loaded GPUs.
4. **Power down cold GPUs** when traffic is low (not all GPUs needed if batch is small
   and only a few experts are active).

```rust
/// Dynamic expert scheduling decisions.
pub enum ExpertScheduleAction {
    /// Replicate an expert to another GPU.
    Replicate { expert_id: ExpertId, target_gpu: GpuId },
    /// Remove a replica (expert underutilized on this GPU).
    Evict { expert_id: ExpertId, source_gpu: GpuId },
    /// Hibernate an entire GPU session (all its experts are cold).
    HibernateGpu { gpu_id: GpuId },
    /// Wake a hibernated GPU session.
    WakeGpu { gpu_id: GpuId },
}
```

This plugs into the existing `SchedulingCostModel` trait from SCHEDULING.md:
the cost model evaluates whether replication/migration reduces expected dispatch
latency more than the memory cost of the replica.

---

## 10. Mac Studio Cluster Considerations

### 10.1 Hardware Profile

| Config | Total Memory | Interconnect | Expert Capacity |
|---|---|---|---|
| 1× M4 Ultra 512GB | 512 GB | Unified (internal) | ~1T FP4 weights |
| 3× M4 Ultra 512GB | 1.5 TB | TB5 80 Gbps | ~2.8T FP4 weights ✅ |
| 4× M4 Ultra 512GB | 2 TB | TB5 80 Gbps | Headroom for KV cache |

### 10.2 Placement Strategy for 3-Node Cluster

```
Mac Studio 0 (512 GB):
  - Experts 0..298 (~1/3 of 896)
  - Attention layers (full — unified memory, no TP needed within node)
  - Router network
  - genai-server orchestrator

Mac Studio 1 (512 GB):
  - Experts 299..597
  - Attention layers (full copy — memory allows it)

Mac Studio 2 (512 GB):
  - Experts 598..895
  - Attention layers (full copy)
```

**Key difference from GPU cluster:** Apple Silicon unified memory means no TP needed
*within* a node. Each Mac Studio can run the full attention layers (they fit in 512 GB
alongside 1/3 of experts). Only expert dispatch crosses nodes.

This dramatically simplifies the architecture: no AllReduce for attention, only
expert All-to-All across TB5.

### 10.3 MLX vs ORT on Apple Silicon

For Mac Studio deployment, MLX may be more appropriate than ORT as the per-node backend:

| | ORT + CoreML EP | MLX |
|---|---|---|
| Unified memory | Through CoreML, less direct | Native, zero-copy |
| Community MoE support | Limited | Growing (mlx-lm supports MoE) |
| Quantization | ONNX quantized models | mlx supports 2/4/8-bit natively |

**Recommendation:** Abstract the per-node execution backend behind a trait so that
the same genai-server orchestration works with ORT sessions (CUDA) or MLX sessions
(Apple Silicon).

```rust
/// Backend-agnostic expert execution session.
#[async_trait]
pub trait ExpertSession: Send + Sync {
    /// Run the designated experts on input hidden states.
    async fn run_experts(
        &self,
        hidden_states: &Tensor,
        expert_ids: &[usize],
    ) -> Result<Tensor>;

    /// Run attention forward pass (for backends that include attention).
    async fn run_attention(
        &self,
        hidden_states: &Tensor,
        kv_cache: &mut KvCacheHandle,
        position: usize,
    ) -> Result<Tensor>;

    /// Device capabilities.
    fn capabilities(&self) -> &MoeEpCapabilities;
}

/// ORT-backed session (CUDA/TensorRT).
pub struct OrtExpertSession { /* ... */ }

/// MLX-backed session (Apple Silicon).
pub struct MlxExpertSession { /* ... */ }
```

---

## 11. Inference Metadata Extensions

The inference metadata spec (onnx/onnx#8184) needs MoE-specific fields:

```yaml
# inference_metadata.yaml — MoE extension
moe:
  num_experts: 896
  top_k: 16
  # Whether there are shared/always-active experts
  shared_experts: 1
  # Router architecture
  router:
    type: "softmax_topk"        # or "sigmoid", "expert_choice", etc.
    aux_loss: "load_balancing"   # routing loss type for reference
    capacity_factor: null        # null = no capacity limit (dropless)
  # Expert FFN architecture
  expert:
    type: "swiglu"              # "gelu", "swiglu", "geglu", etc.
    hidden_dim: 8192
    intermediate_dim: 24576
  # Deployment hints
  deployment:
    min_gpus: 8
    recommended_gpus: 64
    expert_parallel: true
    # Co-activation statistics (optional, from calibration)
    coactivation_stats: "coactivation_matrix.npz"
```

This extends the existing `InferenceMetadata` struct:

```rust
pub struct MoeSpec {
    pub num_experts: usize,
    pub top_k: usize,
    pub shared_experts: usize,
    pub router: RouterSpec,
    pub expert: ExpertArchSpec,
    pub deployment: Option<MoeDeploymentHints>,
}
```

---

## 12. Phased Implementation

### Phase 1: Static EP, Homogeneous GPU (Near-term)

- Contiguous expert placement across N GPUs
- Per-layer dispatch (Option A from §5.2)
- Host-staged communication (PCIe, no CUDA IPC yet)
- Standard MHA/GQA attention with TP AllReduce
- Single-node multi-GPU only
- **Target model:** DeepSeek V3 (671B, 256 experts, top-8) on 8×H200

### Phase 2: Optimized Communication (Mid-term)

- CUDA IPC / DLPack zero-copy for intra-node dispatch
- Per-block dispatch (Option B from §5.2) — fused attention + MoE per session
- Affinity-aware expert placement from calibration data
- Prefix caching that understands MoE routing patterns
- **Target model:** K3-class 2.8T on 16×H200

### Phase 3: Cross-Node + Dynamic (Longer-term)

- NetworkTransport for cross-node dispatch (TB5, RDMA)
- Mac Studio cluster support via MLX backend
- Dynamic expert replication based on runtime activation stats
- GPU hibernation for underutilized expert shards
- Full nxrt EP negotiation integration
- **Target model:** K3-class on 3×M4 Ultra Mac Studio cluster

### Phase 4: Advanced MoE Variants

- LatentMoE / Sigma-MoE (router in latent space)
- Expert choice routing (experts pick tokens, not tokens pick experts)
- Cross-layer expert sharing (same weights, different routing per layer)
- KDA / AttnRes prefix cache support

---

## 13. Open Questions

1. **Router placement:** Should the router run on a dedicated GPU, on CPU, or replicated
   on all GPUs? CPU avoids GPU memory pressure but adds latency. Replicating on all GPUs
   wastes memory but allows local routing decisions.

2. **Combine placement:** The weighted sum of expert outputs — which GPU does it run on?
   Options: (a) dedicated "aggregation" GPU, (b) round-robin, (c) the GPU that will
   run the next attention layer. Option (c) reduces one data movement step.

3. **Shared experts:** K3 and DeepSeek use shared experts that run for every token.
   Should these be replicated on all GPUs (more memory, less communication) or on one
   GPU (less memory, more communication)?

4. **Token dropping:** Under high load, should the dispatcher drop tokens for overloaded
   experts? Or should it queue? MoE capacity factor mechanics from training don't
   directly apply to inference.

5. **ONNX graph structure for expert shards:** One ONNX file per expert shard? Or one
   ONNX file with the full model and runtime-selected subgraph? The former is cleaner
   for deployment; the latter is closer to how models are exported.

6. **Speculative decoding interaction:** With speculative decoding, we generate K draft
   tokens then verify. For MoE, each draft token may route to different experts.
   Should we batch-verify all drafts in one dispatch round? (Probably yes.)

7. **KDA prefix cache:** Kimi contributed a vLLM implementation. Need to study the
   algorithm and determine how it maps to our paged KV cache model.

8. **Quantile Balancing (K3's Stable LatentMoE):** Does the routing algorithm affect
   inference behavior, or is it purely a training concern? If it affects routing scores
   at inference time, we need to replicate the quantile computation.

---

## 14. References

- [Kimi K3 Blog](https://kimi.com/blog/kimi-k3) — 2.8T MoE, KDA, AttnRes, LatentMoE
- [DeepSeek V3 Paper](https://arxiv.org/abs/2412.19437) — 671B MoE, 256 experts, shared experts, MLA
- [Megablocks](https://arxiv.org/abs/2211.15841) — Efficient sparse MoE training/inference kernels
- [Tutel](https://arxiv.org/abs/2206.03382) — Adaptive MoE expert parallelism
- [onnx/onnx#7902](https://github.com/onnx/onnx/issues/7902) — GroupedMatMul ONNX spec (related: MoE expert dispatch)
- [onnx/onnx#8184](https://github.com/onnx/onnx/issues/8184) — Inference Metadata standard
- SCHEDULING.md §8 — nxrt EP Negotiation Protocol
- DESIGN.md §3.2 — KV Cache Manager (paged, tiered)
