# ORT 2.0 — Runtime Design Document

> A Rust-based ONNX runtime built into onnx-genai. First-class plugin EP compatibility,
> strided layout, global-optimal device placement, async data transfer, and exceptional
> debugging — while reusing existing kernels wherever possible.

**Scope:** This document covers the runtime layer (graph IR, execution, memory, EP integration).
The GenAI layer (KV cache, batching, speculative decoding, serving) is covered in [DESIGN.md](./DESIGN.md).

---

## Table of Contents

1. [Design Principles](#1-design-principles)
2. [Architecture Overview](#2-architecture-overview)
3. [Graph IR](#3-graph-ir)
4. [Execution Providers](#4-execution-providers)
5. [Striding and Layout](#5-striding-and-layout)
6. [Cost Model](#6-cost-model)
7. [Graph Partitioning and Device Placement](#7-graph-partitioning-and-device-placement)
8. [Memory Planning](#8-memory-planning)
9. [Async Data Transfer](#9-async-data-transfer)
10. [Dynamic Shape Specialization](#10-dynamic-shape-specialization)
11. [Weight Loading and Storage](#11-weight-loading-and-storage)
12. [Debugging and Profiling](#12-debugging-and-profiling)
13. [Optimization Passes](#13-optimization-passes)
14. [Crate Structure](#14-crate-structure)
15. [Platform Support](#15-platform-support)
16. [Safety and Failure Handling](#16-safety-and-failure-handling)
17. [Open Questions](#17-open-questions)
18. [Phased Roadmap](#18-phased-roadmap)

---

## 1. Design Principles

1. **EP ecosystem is the moat.** Preserve ORT's graph ABI surface so existing plugin EPs
   (QNN, OpenVINO, WebGPU, CoreML, MLX, etc.) work without modification. Any `.so`/`.dylib`
   built against ORT's C provider API loads and runs as-is.

2. **Own the IR.** Internal graph representation inspired by
   [onnx-ir](https://github.com/onnx/ir-py), with strided layout, symbolic dynamic shapes,
   and device placement as first-class concepts. The ONNX protobuf is an import format,
   not the working representation.

3. **Reuse kernels.** CUDA EP and CPU EP are ported from ORT's C++ source. Keep them in C++
   when that's more practical — Rust FFI bindings wrap them. Do not rewrite kernels from
   scratch unless Rust can match performance. Leverage cuDNN, cuBLAS, oneDNN, etc.

4. **Minimize copies.** Strided tensors, layout propagation, and unified memory planning
   eliminate unnecessary data movement. When copies are unavoidable (cross-device), async
   transfer overlaps them with compute.

5. **Cost model drives all decisions.** Device placement, layout transforms, fusion — every
   optimization decision goes through an explicit, inspectable cost model. No hidden heuristics.

6. **Debuggability > cleverness.** Structured cross-device tracing, deterministic replay,
   cost model validation (predicted vs actual), and memory visualization. A runtime that
   can't be debugged can't be trusted.

7. **Global-optimal placement.** Replace ORT's greedy EP-claims-subgraph approach with
   min-cut ILP-based placement that minimizes total cost (compute + transfer).

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                      User API (Rust + C ABI + Python)               │
├─────────────────────────────────────────────────────────────────────┤
│                      Session / InferenceEngine                      │
│                                                                     │
│  ┌───────────┐  ┌────────────────┐  ┌──────────────────────────┐   │
│  │ Cost      │  │ Memory         │  │ Transfer                 │   │
│  │ Model     │  │ Planner        │  │ Scheduler                │   │
│  └─────┬─────┘  └───────┬────────┘  └────────────┬─────────────┘   │
│        │                │                         │                 │
├────────┼────────────────┼─────────────────────────┼─────────────────┤
│        ▼                ▼                         ▼                 │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              Async DAG Executor                              │   │
│  │   (topological schedule, stream-per-device, fence sync)      │   │
│  └────────────────────────┬─────────────────────────────────────┘   │
│                           │                                         │
├───────────────────────────┼─────────────────────────────────────────┤
│              Device / EP Dispatch Layer                              │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────────┐   │
│  │ Native EPs   │  │ Plugin EPs   │  │ Layout / Transfer       │   │
│  │ (CUDA, CPU)  │  │ (QNN, OV,    │  │ Nodes (auto-inserted    │   │
│  │ C++ ported   │  │  WebGPU,     │  │ by placement pass)      │   │
│  │ via FFI      │  │  MLX, ...)   │  │                         │   │
│  └──────────────┘  └──────────────┘  └─────────────────────────┘   │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│              Optimization Passes Pipeline                           │
│  (shape inference → fusion → layout propagation → placement →       │
│   transfer insertion → memory planning)                             │
├─────────────────────────────────────────────────────────────────────┤
│              Graph IR  (strided, symbolic shapes, device-annotated)  │
├─────────────────────────────────────────────────────────────────────┤
│              ONNX Loader  (protobuf + mmap weights → IR)            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Graph IR

### 3.1 Design Goals

- Rich type system: strided layouts, symbolic dims, device placement on every value
- Compatible surface: expose ORT graph ABI to plugin EPs via zero-copy view adapters
- Optimizable: SSA-like structure, immutable after optimization (shared across threads)
- Inspired by [onnx-ir](https://github.com/onnx/ir-py) graph/node/value semantics

### 3.2 Core Types

```rust
/// A value flowing through the graph.
pub struct Value {
    pub id: ValueId,
    pub dtype: DataType,
    pub shape: Shape,
    pub layout: TensorLayout,
    pub device: Option<DeviceId>,  // assigned by placement optimizer
}

/// Shape with static and symbolic dimensions.
pub type Shape = Vec<Dim>;

pub enum Dim {
    Static(usize),
    Symbolic(SymbolId),  // resolved at runtime; carries constraints (min, max, divisor)
}

/// Layout — first-class strides on every value.
pub struct TensorLayout {
    /// Physical strides in elements. None = contiguous row-major.
    pub strides: Option<Vec<i64>>,
    /// Memory format hint for EP compatibility.
    pub format: MemoryFormat,
}

pub enum MemoryFormat {
    Contiguous,       // row-major (C order)
    ChannelsLast,     // NHWC
    Blocked(usize),   // blocked layout (e.g. 16-wide for VNNI/AMX)
    Custom,           // strides encode the full story
}
```

### 3.3 Graph Structure

```rust
pub struct Graph {
    pub nodes: Vec<Node>,
    pub values: Arena<Value>,
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
    pub initializers: HashMap<ValueId, WeightRef>,  // → mmap'd data (§11)
}

pub struct Node {
    pub op_type: String,
    pub domain: String,
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
    pub attributes: HashMap<String, Attribute>,
    pub device: Option<DeviceId>,     // from placement pass
    pub exec_order: Option<usize>,    // from scheduling pass
}
```

### 3.4 ORT Graph ABI Bridge

Plugin EPs see ORT's C API. We provide a **zero-copy view adapter**:

```rust
/// Read-only view that implements ORT's Graph C ABI.
/// No data copy — just function pointers that project our IR.
pub struct OrtGraphView<'a> {
    graph: &'a Graph,
    /// Lazily-built indices matching ORT's query patterns
    node_index: OnceCell<Vec<OrtNodeRepr>>,
}

impl<'a> OrtGraphView<'a> {
    /// Ask a plugin EP what subgraphs it can handle.
    pub fn query_capabilities(&self, ep: &PluginEp) -> Vec<SubgraphClaim>;
    /// Give a plugin EP a subgraph to compile.
    pub fn compile_subgraph(&self, ep: &PluginEp, claim: &SubgraphClaim) -> Result<CompiledKernel>;
}
```

Our IR is the source of truth. Optimization passes operate on our IR.
Plugin EPs get a compatible projection without owning the graph.

---

## 4. Execution Providers

### 4.1 Native EPs (ported from ORT, C++ via FFI)

CUDA EP and CPU EP are ported from ORT upstream. Maintained as C++ with Rust FFI bindings.
They link against cuDNN/cuBLAS/oneDNN directly.

```rust
/// Native EP — we own and maintain the source.
pub trait NativeEp: Send + Sync {
    fn name(&self) -> &str;
    fn device_type(&self) -> DeviceType;

    /// Fine-grained: can this EP run this op with these shapes and input layouts?
    fn supports_op(&self, op: &Node, shapes: &[Shape], layouts: &[TensorLayout]) -> KernelMatch;

    /// Execute a single op.
    fn execute_op(&self, op: &Node, ctx: &mut ExecContext) -> Result<()>;

    /// Execute a fused subgraph (when the fusion pass merged multiple ops).
    fn execute_fused(&self, subgraph: &FusedSubgraph, ctx: &mut ExecContext) -> Result<()>;
}

pub enum KernelMatch {
    /// Can handle. Reports cost and whether it needs a specific layout.
    Supported {
        cost: Cost,
        required_input_layouts: Option<Vec<TensorLayout>>,
        output_layouts: Vec<TensorLayout>,
    },
    Unsupported,
}
```

**Key difference from ORT:** Native EPs report `KernelMatch` with **layout preferences**
and **cost**. The cost model and layout pass use this information for global optimization
rather than local greedy decisions.

### 4.2 Plugin EPs (binary-compatible with ORT C ABI)

QNN, OpenVINO, WebGPU, CoreML, MLX, and any future EPs. Loaded via `dlopen`,
communicate through ORT's C provider API:

```rust
/// Plugin EP — loaded from shared library, speaks ORT C ABI.
pub struct PluginEp {
    lib: Library,
    vtable: OrtProviderVTable,  // function pointers from ORT's C API
    config: EpConfig,
}

impl PluginEp {
    /// Load an existing EP shared library without any modification.
    pub fn load(path: &Path, config: &EpConfig) -> Result<Self>;
}
```

**Compatibility guarantee:** Any EP built against ORT's C provider API works unchanged.
We implement the host side of that ABI. The OrtGraphView (§3.4) bridges our IR to the
EP's expected interface.

### 4.3 Registration and Priority

```rust
pub struct EpRegistry {
    native: Vec<Box<dyn NativeEp>>,
    plugins: Vec<PluginEp>,
    /// Priority order for placement decisions.
    /// The cost model may override this for specific ops.
    priority: Vec<EpId>,
}
```

---

## 5. Striding and Layout

### 5.1 The Problem

ORT assumes contiguous tensors at EP boundaries → forces memcpy for layout changes.
In practice, many ops (elementwise, reductions, normalization) can work with strided input
if they know the strides. Explicit layout transforms (Transpose ops) often exist in the graph
only because the serialization format assumes contiguous storage.

### 5.2 Strided Tensor at Runtime

```rust
/// Runtime tensor — always carries strides.
pub struct Tensor {
    pub buffer: DeviceBuffer,
    pub dtype: DataType,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,     // in elements
    pub offset: usize,           // byte offset into buffer (for views/slices)
}

impl Tensor {
    /// Check if this is standard contiguous (row-major).
    pub fn is_contiguous(&self) -> bool;
    /// Transpose without copy — just reorder strides.
    pub fn transpose(&self, perm: &[usize]) -> Tensor;
    /// Slice without copy — adjust offset, shape, keep strides.
    pub fn slice(&self, dim: usize, range: Range<usize>) -> Tensor;
    /// Make contiguous — copies data only if not already contiguous.
    pub fn contiguous(&self) -> Cow<Tensor>;
}
```

### 5.3 Layout Propagation Pass

Layout is an IR-level annotation. The optimizer propagates preferred layouts through
the graph and only inserts explicit transforms when the cost model says it's cheaper
than letting the downstream kernel handle strided input:

```rust
pub struct LayoutPropagation;

impl OptimizationPass for LayoutPropagation {
    fn run(&self, graph: &mut Graph, cost_model: &CostModel) -> Result<()> {
        // 1. Query each kernel (native EP) for preferred input/output layouts
        // 2. Propagate layouts forward through the graph
        // 3. At conflict points, ask cost model:
        //    cost(insert transform) vs cost(kernel with wrong layout)
        // 4. Insert explicit LayoutTransform nodes only where beneficial
        // 5. Layout-agnostic ops (elementwise, etc.) pass strides through
    }
}
```

**Example — transpose elimination:**
```
Before: MatMul → Transpose → LayerNorm
After:  MatMul(output_strides=transposed) → LayerNorm(reads strided)
        ↑ kernel writes with transposed strides directly, no copy
```

### 5.4 EP Boundary Layout Negotiation

When a plugin EP requires contiguous input (which most legacy EPs do), the placement
pass inserts a `MakeContiguous` node at the boundary — but only for values that are
actually non-contiguous. The cost of this contiguization is factored into the placement
cost model, so the optimizer may choose to keep more ops on the same device to avoid it.

---

## 6. Cost Model

### 6.1 Role

The cost model is the **single source of truth** for all optimization decisions:
placement, layout transforms, fusion, memory tier assignments. No heuristics are
baked into passes — they query the cost model.

### 6.2 Structure

```rust
pub struct CostModel {
    device_profiles: HashMap<DeviceId, DeviceProfile>,
    transfer_matrix: TransferCostMatrix,
    layout_costs: LayoutCostTable,
}

pub struct DeviceProfile {
    /// Peak compute throughput (FLOPS for this dtype)
    pub compute_throughput: HashMap<DataType, f64>,
    /// Memory bandwidth (bytes/sec)
    pub memory_bandwidth: f64,
    /// Kernel launch overhead
    pub launch_overhead: Duration,
    /// Op-specific cost overrides (from profiling or lookup tables)
    pub op_costs: HashMap<OpSignature, OpCost>,
}

/// Transfer cost between any two devices.
pub struct TransferCostMatrix {
    /// cost(src, dst) = latency_base + bytes / bandwidth
    entries: HashMap<(DeviceId, DeviceId), TransferProfile>,
}

pub struct TransferProfile {
    pub latency_base: Duration,    // fixed overhead (PCIe negotiation, DMA setup)
    pub bandwidth: f64,            // bytes/sec sustained
    pub is_async_capable: bool,    // can overlap with compute?
}
```

### 6.3 Queries

```rust
impl CostModel {
    /// Estimated cost of running this op on this device with these layouts.
    pub fn op_cost(&self, op: &Node, device: DeviceId,
                   input_layouts: &[TensorLayout]) -> Cost;

    /// Cost of transforming a tensor from one layout to another on this device.
    pub fn layout_transform_cost(&self, from: &TensorLayout, to: &TensorLayout,
                                  shape: &[usize], device: DeviceId) -> Cost;

    /// Cost of transferring a tensor between devices.
    pub fn transfer_cost(&self, shape: &[usize], dtype: DataType,
                          src: DeviceId, dst: DeviceId) -> TransferCost;

    /// Total estimated cost of a full graph execution plan.
    pub fn total_cost(&self, graph: &Graph, plan: &ExecutionPlan) -> Cost;
}
```

### 6.4 Cost Sources

1. **Static estimates** — lookup tables derived from device specs and op characteristics.
   Good enough for initial planning.
2. **Profiling feedback** — actual measured kernel times feed back to refine the model.
   The profiler (§12) automatically compares predicted vs actual and flags mispredictions.
3. **EP-reported costs** — plugin EPs may optionally report execution time per compiled kernel.

---

## 7. Graph Partitioning and Device Placement

### 7.1 The Problem with ORT's Approach

ORT uses greedy EP claiming: EPs are queried in priority order, each claims subgraphs,
the remainder falls to CPU. This is locally optimal per-EP but globally suboptimal —
it doesn't consider transfer costs between partitions.

### 7.2 Optimal Placement via Min-Cut ILP

We formulate placement as an Integer Linear Program:

```
Minimize:
    Σ compute_cost(node_i, device_i) + Σ transfer_cost(edge_j)

Subject to:
    - Each node assigned to exactly one device
    - Node assigned to device only if that device supports the op
    - Transfer cost for edge = 0 if both endpoints on same device,
      otherwise = cost_model.transfer_cost(tensor_shape, src, dst)
```

```rust
pub struct PlacementOptimizer {
    cost_model: Arc<CostModel>,
    solver: IlpSolver,  // e.g. HiGHS (MIT-licensed LP/ILP solver)
}

impl PlacementOptimizer {
    pub fn optimize(&self, graph: &Graph, eps: &EpRegistry) -> Result<PlacementPlan> {
        // 1. For each node, enumerate feasible devices + compute cost
        // 2. For each edge, compute transfer cost for all src/dst device pairs
        // 3. Solve ILP to minimize total cost
        // 4. Fallback to greedy if ILP is too slow (very large graphs)
    }
}

pub struct PlacementPlan {
    pub node_devices: HashMap<NodeId, DeviceId>,
    pub transfer_edges: Vec<TransferEdge>,  // where to insert copy nodes
}
```

### 7.3 Fallback Strategy

For very large graphs (>10K nodes), ILP may be too slow. Fallback hierarchy:
1. **ILP** — globally optimal (used for graphs up to ~10K nodes)
2. **Min-cut heuristic** — graph partitioning with Kernighan-Lin refinement
3. **Greedy** — ORT-style EP priority claiming (always available, always fast)

The cost model validates the result regardless of which strategy produced it.

### 7.4 Transfer Node Insertion

After placement, the pass inserts explicit `DeviceTransfer` and `MakeContiguous`
nodes at device boundaries:

```rust
pub struct TransferInsertion;

impl OptimizationPass for TransferInsertion {
    fn run(&self, graph: &mut Graph, plan: &PlacementPlan) -> Result<()> {
        for edge in &plan.transfer_edges {
            // Insert: src_device → DeviceTransfer → dst_device
            // If dst EP requires contiguous, also insert MakeContiguous
            // Mark transfer node as async-eligible if both devices support it
        }
    }
}
```

---

## 8. Memory Planning

### 8.1 Lifetime Analysis

After scheduling (topological + placement-aware ordering), analyze when each tensor
is first produced and last consumed:

```rust
pub struct MemoryPlanner;

impl MemoryPlanner {
    pub fn plan(&self, schedule: &ExecSchedule, graph: &Graph) -> MemoryPlan;
}

pub struct MemoryPlan {
    /// Pre-allocated arena per device.
    pub arenas: HashMap<DeviceId, ArenaLayout>,
    /// Tensors that share the same memory (non-overlapping lifetimes).
    pub aliases: Vec<AliasGroup>,
    /// In-place ops (output aliases input buffer).
    pub in_place: Vec<InPlaceOp>,
    /// Persistent allocations (weights, KV cache).
    pub persistent: Vec<PersistentAlloc>,
    /// Staging buffers for async transfers.
    pub staging: Vec<StagingBuffer>,
}
```

### 8.2 Buffer Aliasing

Tensors with non-overlapping lifetimes share physical memory:

```
Execution timeline:
  tensor_a: [====]
  tensor_b:        [======]
  tensor_c: [==]
  → a and b share a buffer (same device, compatible alignment)
  → c and b share a buffer (if sizes match)
```

The planner uses a graph coloring approach on the interference graph
(tensors that are live simultaneously cannot share memory).

### 8.3 In-Place Operations

When an op's output shape matches its input and the input has refcount=1 (no other
consumers), the output can alias the input buffer:

```rust
pub struct InPlaceOp {
    pub node: NodeId,
    pub input_idx: usize,
    pub output_idx: usize,  // output writes directly into input's buffer
}
```

Examples: ReLU, dropout (inference mode), element-wise add (when one input is dead).

### 8.4 Memory Regions

```rust
pub enum MemoryRegion {
    /// Per-inference scratch — allocated at session start, reused across runs.
    Scratch(ArenaId),
    /// Persistent across requests — mmap'd weights, KV cache pages.
    Persistent(PersistentPoolId),
    /// Staging buffers for async DMA (pinned host memory).
    TransferStaging(StagingBufferId),
}
```

---

## 9. Async Data Transfer

### 9.1 The Problem

Synchronous memcpy stalls the entire pipeline. GPU→CPU copy blocks the GPU compute stream.
Even when transfers are unavoidable, they should overlap with independent computation.

### 9.2 Transfer Scheduler

```rust
pub struct TransferScheduler {
    /// Per-device transfer streams (separate from compute streams).
    streams: HashMap<DeviceId, TransferStream>,
    /// Pending transfers with fence synchronization.
    pending: Vec<PendingTransfer>,
}

pub struct PendingTransfer {
    pub src_buffer: DeviceBuffer,
    pub dst_buffer: DeviceBuffer,
    pub fence: Fence,            // signaled when transfer completes
    pub needed_by: NodeId,       // which compute node depends on this
    pub issued_after: NodeId,    // which compute node produced the source data
}
```

### 9.3 Overlap Strategy

The DAG executor schedules transfers **as early as possible** (once source data is ready)
and dependent compute **as late as possible** (just after the fence resolves):

```
Timeline (GPU + CPU):

GPU compute:  [Attention_L1]  [Attention_L2]  [Attention_L3] ...
              ─────────────────────────────────────────────────
GPU→CPU xfer:      [xfer logits]─────────┐
              ─────────────────────────────────────────────────
CPU compute:                              └──[Sampling]──[Token selection]

→ Transfer of L1 logits overlaps with L2 attention. Zero bubble.
```

### 9.4 Double Buffering

For token-by-token LLM generation, alternate between two staging buffers:

```rust
pub struct DoubleBuffer {
    buffers: [StagingBuffer; 2],
    current: AtomicUsize,
}

impl DoubleBuffer {
    /// Swap: current buffer becomes the "in-flight transfer" buffer,
    /// the other becomes the "next write" buffer.
    pub fn swap(&self) -> &StagingBuffer;
}
```

While buffer A is transferring step N's results, step N+1 writes into buffer B on GPU.
No synchronization between the two.

### 9.5 Prefetch

For tiered KV cache (GPU → CPU eviction), the scheduler can prefetch pages back
to GPU ahead of when they're needed:

```rust
impl TransferScheduler {
    /// Issue prefetch of KV pages from CPU→GPU, anticipating upcoming attention.
    pub fn prefetch_kv(&mut self, pages: &[PageId], target: DeviceId) -> Fence;
}
```

---

## 10. Dynamic Shape Specialization

### 10.1 The Problem

Some EPs (TensorRT, QNN) compile kernels for specific tensor shapes. When shapes change
at runtime (different sequence lengths, batch sizes), compiled kernels are invalid.

### 10.2 Shape-Keyed Kernel Cache

```rust
pub struct KernelCache {
    /// Compiled kernels keyed by (EP, subgraph_hash, concrete_shapes).
    entries: HashMap<KernelCacheKey, CompiledKernel>,
    /// Budget: max entries per EP (LRU eviction).
    max_entries_per_ep: usize,
}

pub struct KernelCacheKey {
    pub ep_id: EpId,
    pub subgraph_hash: u64,
    pub shapes: Vec<Vec<usize>>,  // concrete input shapes
}
```

### 10.3 Compilation Strategies

```rust
pub enum ShapeStrategy {
    /// Compile once with symbolic shapes (if EP supports it).
    /// Most flexible, may sacrifice performance.
    Symbolic,

    /// Compile for common shapes upfront, JIT for new shapes.
    /// Good for LLM: prefill (variable seq_len) + decode (seq_len=1) are the two hot shapes.
    CommonShapes {
        warmup_shapes: Vec<Vec<usize>>,
    },

    /// Compile on first encounter, cache for reuse.
    /// Adds latency on first run with new shape.
    JitAndCache,

    /// Bucket shapes into size classes (e.g. seq_len rounded up to next power of 2).
    /// Reduces cache entries at the cost of some padding.
    Bucketed {
        bucket_fn: Box<dyn Fn(&[usize]) -> Vec<usize>>,
    },
}
```

### 10.4 Compilation Cost Budgeting

Recompilation can be expensive (TensorRT: seconds). The cost model accounts for this:

```rust
impl CostModel {
    /// Is it cheaper to recompile for exact shapes, use a bucketed kernel,
    /// or fall back to a shape-generic kernel?
    pub fn recompilation_decision(
        &self, ep: EpId, subgraph: &SubgraphView,
        current_shapes: &[Vec<usize>], cached: &KernelCache,
    ) -> RecompileDecision;
}

pub enum RecompileDecision {
    ReuseExact(KernelCacheKey),
    ReuseBucketed(KernelCacheKey),
    Recompile,
    FallbackToGeneric,
}
```

---

## 11. Weight Loading and Storage

### 11.1 Memory-Mapped Weights

Weights are loaded via **mmap**, not read into heap buffers:

```rust
pub struct WeightStore {
    /// mmap'd regions for each weight file.
    mappings: Vec<Mmap>,
    /// Index: weight name → (mapping_idx, byte_offset, byte_length, dtype, shape).
    index: HashMap<String, WeightRef>,
}

pub struct WeightRef {
    pub mapping_idx: usize,
    pub offset: usize,
    pub length: usize,
    pub dtype: DataType,
    pub shape: Vec<usize>,
}

impl WeightStore {
    /// Get a tensor view into mmap'd data. Zero-copy.
    pub fn get(&self, name: &str) -> Option<TensorView<'_>>;
}
```

**Benefits:**
- Zero-copy weight loading — OS pages in on demand
- Multiple sessions sharing the same model share physical memory (OS dedup)
- Lazy loading — unused weights (e.g. in MoE, only active experts) never page in

### 11.2 Supported Formats

```rust
pub enum WeightFormat {
    /// ONNX external data (raw tensor files, referenced by protobuf).
    OnnxExternal,
    /// Safetensors — header + raw tensors, mmap-friendly.
    Safetensors,
    /// GGUF — quantized format used by llama.cpp ecosystem.
    Gguf,
}
```

All formats are loaded through a unified `WeightStore` interface. Format detection is
automatic based on file extension and magic bytes.

### 11.3 Weight Sharing Across Sessions

Multiple `InferenceSession`s running the same model share the same `WeightStore`
(and therefore the same mmap'd physical pages). Only mutable state (KV cache, scratch
arena) is per-session.

```rust
pub struct ModelInstance {
    pub graph: Arc<Graph>,           // immutable after optimization
    pub weights: Arc<WeightStore>,   // shared mmap
}

pub struct InferenceSession {
    pub model: Arc<ModelInstance>,    // shared
    pub scratch: Arena,              // per-session
    pub kv_cache: KvCacheManager,    // per-session (from onnx-genai)
}
```

---

## 12. Debugging and Profiling

### 12.1 Design Goals

- **Zero overhead when off** — no cost in release builds unless explicitly enabled
- **Cross-device unified timeline** — CPU, GPU, NPU, and transfers on one timeline
- **Cost model validation** — automatic comparison of predicted vs actual times
- **Memory visualization** — see allocations, aliasing, fragmentation, peak usage
- **Deterministic replay** — capture all inputs + random state to reproduce any run

### 12.2 Trace Spans

```rust
/// Every op execution emits a trace span (when profiling is enabled).
pub struct TraceSpan {
    pub name: String,
    pub node_id: NodeId,
    pub device: DeviceId,
    pub stream: StreamId,           // compute or transfer stream
    pub start_ns: u64,
    pub end_ns: u64,
    pub memory_allocated: usize,
    pub memory_freed: usize,
    pub input_shapes: Vec<Vec<usize>>,
    pub predicted_cost: Option<Cost>,   // what cost model estimated
    pub actual_cost: Option<Cost>,      // measured (when available)
}
```

### 12.3 Export Formats

```rust
pub enum TraceExportFormat {
    /// Chrome Trace JSON (chrome://tracing)
    ChromeTrace,
    /// Perfetto protobuf (cross-process, cross-device visualization)
    Perfetto,
    /// JSON Lines (for programmatic analysis)
    JsonLines,
}
```

### 12.4 Profiling Modes

```rust
pub enum ProfilingMode {
    Off,
    /// Per-op wall time only (minimal overhead).
    Timing,
    /// Timing + memory + transfers + cost model comparison.
    Full,
    /// Emit spans to a tracing subscriber (e.g. `tracing` crate).
    Tracing { subscriber: Box<dyn TraceSubscriber> },
}
```

### 12.5 Memory Debugger

```rust
pub struct MemoryDebugger {
    /// Dump a map of all allocations on a device at this instant.
    pub fn allocation_map(&self, device: DeviceId) -> AllocationMap;
    /// Peak usage over time (for visualization).
    pub fn peak_timeline(&self) -> Vec<(Timestamp, usize)>;
    /// Detect leaked buffers (allocated but never freed after session ends).
    pub fn detect_leaks(&self) -> Vec<LeakedBuffer>;
    /// Validate that the memory plan's aliasing is correct (no overlap violations).
    pub fn validate_aliasing(&self, plan: &MemoryPlan) -> Result<()>;
}
```

### 12.6 Graph Dump

Dump the IR at any optimization stage with placement, layout, and cost annotations:

```rust
pub fn dump_graph(graph: &Graph, stage: &str, format: GraphDumpFormat) -> String;

pub enum GraphDumpFormat {
    Dot,        // Graphviz — with device coloring and layout annotations
    Json,       // for custom viewers
    OnnxProto,  // re-export as ONNX for Netron
}
```

### 12.7 Cost Model Misprediction Report

After profiled execution, automatically flag where the cost model was significantly wrong:

```rust
pub struct MispredictionReport {
    pub entries: Vec<MispredictionEntry>,
}

pub struct MispredictionEntry {
    pub node_id: NodeId,
    pub op: String,
    pub device: DeviceId,
    pub predicted_us: f64,
    pub actual_us: f64,
    pub ratio: f64,  // actual / predicted. >2.0 or <0.5 = flagged
}
```

---

## 13. Optimization Passes

Ordered pipeline. Each pass operates on the IR and may query the cost model:

```rust
pub fn default_passes() -> Vec<Box<dyn OptimizationPass>> {
    vec![
        // 1. Graph normalization
        Box::new(ConstantFolding),
        Box::new(ShapeInference),
        Box::new(DeadNodeElimination),

        // 2. Op fusion (before placement — fused ops may have different device affinity)
        Box::new(OpFusion),           // MatMul+Bias+Relu → FusedGemm
        Box::new(AttentionFusion),    // Q/K/V/Softmax/V pattern → FlashAttention node

        // 3. Layout and placement (cost-model-driven)
        Box::new(LayoutPropagation),     // propagate preferred layouts, eliminate transposes
        Box::new(PlacementOptimizer),    // assign ops to devices (ILP/min-cut/greedy)
        Box::new(TransferInsertion),     // insert async copy nodes at device boundaries
        Box::new(MakeContiguousInsertion), // insert contiguization where EPs require it

        // 4. Memory
        Box::new(InPlaceDetection),      // mark ops that can alias input↔output
        Box::new(MemoryPlanning),        // lifetime analysis, arena sizing, buffer aliasing
        Box::new(StagingBufferAlloc),    // allocate pinned buffers for async transfers
    ]
}
```

---

## 14. Crate Structure

```
onnx-genai/                              (monorepo — was onnx-genai, now includes runtime)
├── crates/
│   │
│   │  ── Runtime layer (new) ──
│   ├── ort-ir/                          # Graph IR, types, shapes, strides, layout
│   ├── ort-loader/                      # ONNX protobuf → IR, weight mmap
│   ├── ort-optimizer/                   # Optimization passes pipeline
│   ├── ort-cost-model/                  # Cost estimation, profiling feedback loop
│   ├── ort-memory/                      # Memory planner, arenas, aliasing, staging
│   ├── ort-scheduler/                   # Async DAG executor, transfer scheduler
│   ├── ort-ep-api/                      # EP trait defs + ORT C ABI bridge
│   ├── ort-ep-cpu/                      # CPU EP (C++ ported from ORT, Rust FFI)
│   ├── ort-ep-cuda/                     # CUDA EP (C++ ported from ORT, Rust FFI)
│   ├── ort-profiler/                    # Tracing, Chrome Trace/Perfetto, memory debugger
│   ├── ort-session/                     # Session management, model loading, inference API
│   │
│   │  ── GenAI layer (existing) ──
│   ├── onnx-genai-kv/                  # KV cache (paged, tiered, heterogeneous heads)
│   ├── onnx-genai-engine/              # Batching, speculative, pipeline
│   ├── onnx-genai-server/              # OpenAI-compatible HTTP API
│   ├── onnx-genai-router/              # Multi-node routing
│   ├── onnx-genai-ort/                 # ORT C API bindings (→ replaced by ort-session)
│   ├── onnx-genai-metadata/            # inference_metadata.yaml schema
│   └── ...
│
├── native-eps/
│   ├── cpu/                             # C++ CPU kernels (ported from ORT)
│   └── cuda/                            # C++ CUDA kernels (ported from ORT)
│
├── bindings/
│   ├── python/                          # PyO3 bindings
│   └── c-api/                           # C ABI for external consumers
│
├── docs/
│   ├── DESIGN.md                        # GenAI layer design (existing)
│   ├── ORT2.md                          # This document — runtime design
│   └── PROGRESS.md
│
└── Cargo.toml                           # Workspace
```

**Migration path:** The GenAI crates (`onnx-genai-engine`, `onnx-genai-kv`, etc.) depend on
a **backend trait**, not a concrete runtime. At build time, a Cargo feature selects which
backend to compile against:

```toml
# In onnx-genai-engine/Cargo.toml
[features]
default = ["backend-ort"]        # use upstream ORT via C API bindings
backend-ort = ["dep:onnx-genai-ort"]   # existing ORT C API wrapper
backend-ort2 = ["dep:ort-session"]     # our own runtime
```

```rust
/// Backend-agnostic inference trait.
/// Both onnx-genai-ort (upstream ORT) and ort-session (ORT 2.0) implement this.
pub trait InferenceBackend: Send + Sync {
    type Session: InferenceSession;
    fn load_model(&self, path: &Path, options: &SessionOptions) -> Result<Self::Session>;
}

pub trait InferenceSession: Send {
    fn run(&mut self, inputs: &[Tensor], outputs: &mut [Tensor]) -> Result<()>;
    fn io_binding(&mut self) -> Result<IoBinding<'_>>;
}
```

This lets users choose:
- `backend-ort`: production-proven, full opset coverage, all existing EPs — the safe default
- `backend-ort2`: our runtime with better placement, async transfer, strided layout — opt-in

Both backends are tested in CI. The GenAI layer is backend-agnostic — KV cache, batching,
speculative decoding, and the HTTP server work identically regardless of which runtime
executes the ONNX graph underneath.

---

## 15. Platform Support

| Platform | Native EPs | Plugin EPs | Weight mmap | Notes |
|----------|-----------|------------|-------------|-------|
| Linux x64 | CUDA, CPU | QNN, OpenVINO, WebGPU | ✅ | Primary target |
| macOS arm64 | CPU | MLX, CoreML, WebGPU | ✅ | MLX for Apple Silicon GPU |
| Windows x64 | CUDA, CPU | QNN, OpenVINO, WebGPU | ✅ | |
| Linux arm64 | CPU | QNN, WebGPU | ✅ | Edge / mobile |
| Web (WASM) | — | WebGPU | ❌ (fetch) | Via wasm-bindgen |

---

## 16. Safety and Failure Handling

### 16.1 FFI Boundary Safety

All C/C++ FFI calls (native EP kernels, plugin EP dlopen) are wrapped with:

```rust
/// Catch panics and C++ exceptions at the FFI boundary.
/// Never let a plugin EP crash the host process.
pub fn safe_ffi_call<F, T>(f: F) -> Result<T>
where F: FnOnce() -> T + UnwindSafe {
    std::panic::catch_unwind(f)
        .map_err(|_| Error::EpPanicked)
}
```

### 16.2 EP Fallback

If an EP fails (crash, OOM, unsupported shape at runtime):
1. Log the failure with full context (spans, inputs, shapes)
2. Mark the failing kernel/subgraph as "poisoned" for this EP
3. Re-plan: remove the failed EP from the placement, re-run placement optimizer
4. Continue execution on fallback device

### 16.3 Thread Safety Model

- **Graph IR:** immutable after optimization → shared freely (`Arc<Graph>`)
- **Weights:** read-only mmap → shared (`Arc<WeightStore>`)
- **Execution state:** owned by session, single-writer (no locks needed)
- **EP calls:** serialized per-EP (some EPs are not thread-safe)
- **Transfer scheduler:** owns its streams, communicates with executor via channels

---

## 17. Open Questions

1. **ONNX Runtime Extensions (custom ops)** — Support via the same C ABI bridge?
   Likely yes, since they use the same registration mechanism.

2. **Quantization story** — Runtime quantization (KV cache fp8, activation quantization)
   is handled in the GenAI layer. Pre-quantized model weights (INT4/INT8) need kernel support
   in native EPs. What about mixed-precision inference (different layers at different precision)?

3. **JIT compilation** — Should we integrate a JIT backend (e.g. Cranelift, LLVM) for
   generating fused kernels at runtime? Or leave this to specialized EPs (TensorRT, XLA)?

4. **Model format** — Should we define our own optimized model format (IR + weights in one file)
   for faster loading, or always load from ONNX protobuf + external weights?

5. **Backwards compat** — Which ONNX opset versions do we support? Minimum viable: opset 17+
   (covers all modern LLMs). Full ORT compat would require opset 7+.

---

## 18. Phased Roadmap

### Phase 1: IR + Loader + Single-Device Execution
- [ ] Graph IR with strided layout support
- [ ] ONNX protobuf → IR loader
- [ ] Weight mmap (safetensors, ONNX external data)
- [ ] CPU EP (subset of ops, ported from ORT C++)
- [ ] Sequential executor (no async)
- [ ] Basic session API (load, run)
- [ ] Per-op timing profiler

### Phase 2: Multi-Device + Plugin EP Compatibility
- [ ] ORT Graph ABI bridge (OrtGraphView)
- [ ] Plugin EP loading (dlopen + vtable)
- [ ] CUDA EP integration (C++ FFI)
- [ ] Cost model (static estimates)
- [ ] Placement optimizer (start with greedy, then ILP)
- [ ] Transfer node insertion
- [ ] Async transfer scheduler
- [ ] Layout propagation pass

### Phase 3: Memory + Performance
- [ ] Memory planner (lifetime analysis, aliasing, in-place)
- [ ] Double-buffered async transfers
- [ ] Op fusion passes (Gemm, Attention)
- [ ] Cross-device Chrome Trace / Perfetto profiling
- [ ] Cost model profiling feedback loop
- [ ] Dynamic shape kernel cache
- [ ] Shape bucketing for compilation EPs

### Phase 4: GenAI Integration
- [ ] Replace onnx-genai-ort bindings with ort-session
- [ ] KV cache on ort-memory arenas
- [ ] Continuous batching through ort-scheduler
- [ ] End-to-end: ONNX model → GenAI server, no external ORT dependency

### Phase 5: Production Hardening
- [ ] Python bindings (onnxruntime-compatible API surface)
- [ ] C ABI for external consumers
- [ ] Opset coverage: top 50 models on HuggingFace
- [ ] Benchmark suite + CI regression tracking
- [ ] Memory debugger + leak detection
- [ ] Plugin EP development guide
- [ ] GGUF weight loading
