# nxrt — Runtime Design Document

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
10. [Compute-Communication Overlap](#10-compute-communication-overlap)
11. [Dynamic Shape Specialization](#11-dynamic-shape-specialization)
12. [Weight Loading and Storage](#12-weight-loading-and-storage)
13. [Flash Attention Integration](#13-flash-attention-integration)
14. [CUDA Graph Capture](#14-cuda-graph-capture)
15. [CUDA EP Kernel Strategy](#15-cuda-ep-kernel-strategy)
16. [Auto-Tuning Agent Interface](#16-auto-tuning-agent-interface)
17. [Debugging and Profiling](#17-debugging-and-profiling)
18. [Optimization Passes](#18-optimization-passes)
19. [ONNX Loader](#19-onnx-loader)
20. [Session API](#20-session-api)
21. [ORT C API Compatibility](#21-ort-c-api-compatibility)
22. [Error Types](#22-error-types)
23. [Crate Structure](#23-crate-structure)
24. [Python Bindings](#24-python-bindings)
25. [Platform Support](#25-platform-support)
26. [Safety and Failure Handling](#26-safety-and-failure-handling)
27. [Testing Strategy](#27-testing-strategy)
28. [Open Questions](#28-open-questions)
29. [Phased Roadmap](#29-phased-roadmap)

<!-- NOTE: entries 1–29 above predate later sections; the body currently runs to §58.
     Full TOC reconciliation is tracked separately. New sections are appended below. -->

55. [EPContext Node — On-Disk Compiled-EP Interchange](#55-epcontext-node--on-disk-compiled-ep-interchange)
56. [Phased Roadmap](#56-phased-roadmap)

---

## 1. Design Principles

1. **EP ecosystem is the moat.** Preserve ORT's graph ABI surface so existing plugin EPs
   (QNN, OpenVINO, WebGPU, CoreML, MLX, etc.) work without modification.

2. **All EPs are plugins.** Every EP (including CUDA, CPU) is an independent crate with
   a uniform plugin interface. They work with both ORT (via C ABI export) and our runtime
   (via native Rust trait). snake_case naming to distinguish from ORT's PascalCase EPs.

3. **Own the IR.** Internal graph representation inspired by onnx-ir, with strided layout,
   symbolic dynamic shapes, and device placement as first-class concepts.

4. **Reuse kernels.** CUDA EP uses CuTe (CUTLASS 3.x) for high-frequency ops and calls
   cuBLAS/cuDNN for battle-tested paths. CPU EP uses oneDNN. Don't rewrite what works.

5. **Minimize copies.** Strided tensors, layout propagation, and async transfer overlap
   eliminate unnecessary data movement.

6. **Cost model drives all decisions.** Device placement, layout transforms, fusion — every
   optimization goes through an explicit, inspectable cost model. No hidden heuristics.

7. **Global-optimal placement.** Min-cut ILP replaces ORT's greedy EP-claims-subgraph.

8. **Debuggability > cleverness.** Cross-device tracing, deterministic replay, cost model
   validation, memory visualization.

9. **Agent-optimizable.** An LLM agent can profile, analyze, suggest, and apply runtime
   tuning in a closed loop.

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
│  ┌─────▼─────┐  ┌───────▼────────┐  ┌────────────▼─────────────┐   │
│  │ Auto-     │  │ Arena          │  │ Stream Manager           │   │
│  │ Tuner     │  │ Allocator      │  │ (CUDA/Metal streams)     │   │
│  └───────────┘  └────────────────┘  └──────────────────────────┘   │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│              Async DAG Executor                                      │
│   (topological schedule, per-device streams, fence sync,            │
│    CUDA graph capture, compute-comm overlap)                        │
├─────────────────────────────────────────────────────────────────────┤
│              EP Dispatch Layer                                       │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────────┐   │
│  │ ort_ep_cpu   │  │ ort_ep_cuda  │  │ Plugin EPs              │   │
│  │ (our crate)  │  │ (CuTe+cuBLAS)│  │ (QNN, OV, WebGPU,      │   │
│  │              │  │              │  │  MLX, CoreML, ROCm)     │   │
│  └──────────────┘  └──────────────┘  └─────────────────────────┘   │
│         ▲                  ▲                    ▲                    │
│         └──────────────────┴────────────────────┘                   │
│              All EPs implement ExecutionProvider trait               │
│              + export_ort_plugin!() for ORT compatibility            │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│              Optimization Passes Pipeline                           │
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
- Optimizable: SSA-like, immutable after optimization (shared across threads via `Arc`)
- Support subgraphs (control flow: If, Loop, Scan)

**Reference implementation:** [onnx-ir](https://github.com/onnx/ir-py) (`pip install onnx-ir`)

The Python `onnx-ir` package (authored by Justin) defines the canonical IR design we port to Rust.
Key concepts to preserve:

| onnx-ir (Python) | onnx-runtime-ir (Rust) | Notes |
|------------------|------------------------|-------|
| `ir.Graph` | `Graph` | Node arena + value arena + I/O lists |
| `ir.Node` | `Node` | Op + inputs (optional) + outputs + attrs |
| `ir.Value` | `Value` | Typed, shaped, tracks producer/consumers |
| `ir.Attr` / `ir.RefAttr` | `Attribute` | All ONNX attr types |
| `ir.Tensor` / `ir.ExternalTensor` | `WeightRef` + mmap | Lazy-loaded, memory-mapped |
| `ir.TypeProto` / `ir.Type` | `DataType` + `Shape` | Split into concrete types |
| `ir.Shape` with symbolic dims | `Shape` = `Vec<Dim>` | `Dim::Symbolic(SymbolId)` |
| `ir.Graph.topological_sort()` | `Graph::topological_order()` | Kahn's algorithm |
| `ir.passes.*` | `onnx-runtime-optimizer` passes | Separate crate |
| `ir.traversal` | `Graph::predecessors/successors` | Graph query API |

**What we add beyond onnx-ir:**
- Strided `TensorLayout` on every value (onnx-ir doesn't track physical layout)
- `DeviceId` placement annotation (for multi-device)
- Mutation API for optimization passes (onnx-ir is mostly immutable)
- ORT Graph ABI bridge (C-compatible projection for plugin EPs)
- Memory format / alignment annotations

**What we intentionally don't port:**
- Python-specific conveniences (e.g. `__repr__`, numpy interop)
- ONNX checker/validator (we have our own `Graph::validate()`)
- Serialization back to protobuf (we only load, never save ONNX)

### 3.2 Core Types

```rust
/// Unique identifier for values in the graph.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ValueId(pub u32);

/// Unique identifier for nodes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct NodeId(pub u32);

/// Unique identifier for symbolic dimensions.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SymbolId(pub u32);

/// A value flowing through the graph.
pub struct Value {
    pub id: ValueId,
    pub name: Option<String>,
    pub dtype: DataType,
    pub shape: Shape,
    pub layout: TensorLayout,
    pub device: Option<DeviceId>,
    pub producer: Option<NodeId>,
    pub consumers: Vec<NodeId>,
}

/// Supported data types (matching ONNX TensorProto.DataType).
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum DataType {
    Float32 = 1, Uint8 = 2, Int8 = 3, Uint16 = 4, Int16 = 5,
    Int32 = 6, Int64 = 7, String = 8, Bool = 9, Float16 = 10,
    Float64 = 11, Uint32 = 12, Uint64 = 13, BFloat16 = 16,
    Float8E4M3FN = 17, Float8E5M2 = 18, Int4 = 22, Uint4 = 23,
}

impl DataType {
    /// Byte size per element. Sub-byte types return 0.
    pub fn byte_size(&self) -> usize {
        match self {
            Self::Float32 | Self::Int32 | Self::Uint32 => 4,
            Self::Float64 | Self::Int64 | Self::Uint64 => 8,
            Self::Float16 | Self::BFloat16 | Self::Int16 | Self::Uint16 => 2,
            Self::Int8 | Self::Uint8 | Self::Bool | Self::Float8E4M3FN | Self::Float8E5M2 => 1,
            Self::Int4 | Self::Uint4 => 0, // packed, 2 per byte
            Self::String => 0,
        }
    }
    pub fn bit_size(&self) -> usize {
        match self {
            Self::Int4 | Self::Uint4 => 4,
            other => other.byte_size() * 8,
        }
    }
    pub fn is_float(&self) -> bool {
        matches!(self, Self::Float32 | Self::Float64 | Self::Float16
            | Self::BFloat16 | Self::Float8E4M3FN | Self::Float8E5M2)
    }
}

/// Shape with static and symbolic dimensions.
pub type Shape = Vec<Dim>;

#[derive(Clone, Debug, PartialEq)]
pub enum Dim {
    Static(usize),
    Symbolic(SymbolId),
}

/// Constraints on symbolic dimensions.
pub struct SymbolConstraints {
    pub id: SymbolId,
    pub name: Option<String>,      // e.g. "batch_size", "seq_len"
    pub min: Option<usize>,        // minimum value
    pub max: Option<usize>,        // maximum value
    pub divisible_by: Option<usize>,  // must be multiple of (for tiling)
}

/// Layout — first-class strides on every value.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorLayout {
    /// Physical strides in elements. None = contiguous row-major.
    pub strides: Option<Vec<i64>>,
    /// Memory format hint.
    pub format: MemoryFormat,
    /// Alignment requirement in bytes.
    pub alignment: usize,
}

impl TensorLayout {
    /// Create contiguous layout for given shape.
    pub fn contiguous(shape: &[usize]) -> Self {
        Self { strides: None, format: MemoryFormat::Contiguous, alignment: 64 }
    }

    /// Check if actual strides match contiguous.
    pub fn is_contiguous(&self, shape: &[usize]) -> bool {
        match &self.strides {
            None => true,
            Some(s) => {
                let expected = Self::compute_contiguous_strides(shape);
                s == &expected
            }
        }
    }

    /// Compute contiguous strides for a shape (row-major).
    pub fn compute_contiguous_strides(shape: &[usize]) -> Vec<i64> {
        let mut strides = vec![1i64; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1] as i64;
        }
        strides
    }

    /// Transpose: reorder strides without copying data.
    pub fn transpose(&self, shape: &[usize], perm: &[usize]) -> Self {
        let strides = self.strides.as_ref()
            .map(|s| perm.iter().map(|&p| s[p]).collect())
            .unwrap_or_else(|| {
                let cs = Self::compute_contiguous_strides(shape);
                perm.iter().map(|&p| cs[p]).collect()
            });
        Self { strides: Some(strides), format: MemoryFormat::Custom, alignment: self.alignment }
    }

    /// Total storage size in bytes (max offset reachable by strides).
    pub fn storage_size(&self, shape: &[usize], dtype: DataType) -> usize {
        let elem_size = dtype.byte_size().max(1);
        match &self.strides {
            None => shape.iter().product::<usize>() * elem_size,
            Some(strides) => {
                let max_offset: i64 = shape.iter().zip(strides.iter())
                    .map(|(&dim, &stride)| (dim.saturating_sub(1)) as i64 * stride.abs())
                    .sum();
                (max_offset as usize + 1) * elem_size
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum MemoryFormat {
    Contiguous,
    ChannelsLast,    // NHWC
    Blocked(usize),  // e.g. 16-wide for VNNI/AMX
    Custom,
}
```

### 3.3 Graph Structure

```rust
pub struct Graph {
    pub nodes: Arena<Node>,
    pub values: Arena<Value>,
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
    pub initializers: HashMap<ValueId, WeightRef>,
    pub symbol_constraints: HashMap<SymbolId, SymbolConstraints>,
    pub opset_imports: HashMap<String, u64>,  // domain → version
    pub subgraphs: HashMap<(NodeId, String), Graph>,  // (node, attr_name) → subgraph
}

impl Graph {
    // === Query API ===

    pub fn node(&self, id: NodeId) -> &Node;
    pub fn node_mut(&mut self, id: NodeId) -> &mut Node;
    pub fn value(&self, id: ValueId) -> &Value;
    pub fn value_mut(&mut self, id: ValueId) -> &mut Value;
    pub fn num_nodes(&self) -> usize;
    pub fn num_values(&self) -> usize;

    /// Topological order via Kahn's algorithm. Returns error if cycle detected.
    pub fn topological_order(&self) -> Result<Vec<NodeId>, GraphError> {
        // 1. Compute in-degree for each node
        // 2. Queue all nodes with in-degree 0
        // 3. Pop from queue, add to order, decrement successors' in-degree
        // 4. If order.len() != num_nodes → cycle
        todo!()
    }

    /// Direct predecessors: nodes that produce this node's inputs.
    pub fn predecessors(&self, node: NodeId) -> Vec<NodeId> {
        self.node(node).inputs.iter()
            .filter_map(|opt| opt.as_ref())
            .filter_map(|vid| self.value(*vid).producer)
            .collect()
    }

    /// Direct successors: nodes that consume this node's outputs.
    pub fn successors(&self, node: NodeId) -> Vec<NodeId> {
        self.node(node).outputs.iter()
            .flat_map(|vid| self.value(*vid).consumers.iter().copied())
            .collect()
    }

    /// All nodes between two sets (subgraph extraction for EP claims).
    pub fn nodes_between(&self, inputs: &[ValueId], outputs: &[ValueId]) -> Vec<NodeId>;

    // === Mutation API (optimization passes use these) ===

    /// Insert a new node. Updates value producer/consumer links.
    pub fn insert_node(&mut self, node: Node) -> NodeId;

    /// Remove a node. Disconnects edges. Orphaned output values are deleted.
    pub fn remove_node(&mut self, id: NodeId);

    /// Replace a node in-place, rewiring its input/output edges to the new node.
    pub fn replace_node(&mut self, old: NodeId, new: Node) -> NodeId;

    /// Insert a node on an edge: producer → [new_node] → consumer.
    /// The new node's single input is the original value,
    /// and it produces a new value that replaces the original in all consumers.
    pub fn insert_on_edge(&mut self, value: ValueId, new_node: Node) -> NodeId;

    /// Replace all uses of `old_value` with `new_value` in consumer nodes.
    pub fn replace_all_uses(&mut self, old_value: ValueId, new_value: ValueId);

    /// Create a new value (e.g. for inserted nodes).
    pub fn create_value(&mut self, dtype: DataType, shape: Shape) -> ValueId;

    // === Validation ===

    /// Verify graph invariants. Call after optimization passes in debug builds.
    pub fn validate(&self) -> Result<(), Vec<GraphError>> {
        // 1. Every value's producer exists and has it in outputs
        // 2. Every value's consumers exist and have it in inputs
        // 3. No dangling ValueIds in node inputs/outputs
        // 4. Graph inputs have no producer
        // 5. Graph outputs have a producer
        // 6. No cycles (topological_order succeeds)
        // 7. Opset imports are valid
        // 8. Subgraphs validate recursively
        todo!()
    }
}

pub struct Node {
    pub id: NodeId,
    pub op_type: String,
    pub domain: String,
    pub inputs: Vec<Option<ValueId>>,  // Option for optional inputs
    pub outputs: Vec<ValueId>,
    pub attributes: HashMap<String, Attribute>,
    pub doc_string: Option<String>,
    pub device: Option<DeviceId>,
    pub exec_order: Option<usize>,
}

/// ONNX attribute types.
#[derive(Clone, Debug)]
pub enum Attribute {
    Int(i64),
    Float(f32),
    String(String),
    Ints(Vec<i64>),
    Floats(Vec<f32>),
    Strings(Vec<String>),
    Tensor(TensorData),
    Graph(Box<Graph>),
    Graphs(Vec<Graph>),
    SparseTensor(SparseTensorData),
    TypeProto(TypeProto),
}

#[derive(Clone, Debug)]
pub enum GraphError {
    DanglingValue(ValueId),
    DanglingNode(NodeId),
    CycleDetected,
    MissingProducer(ValueId),
    DuplicateOutput(ValueId),
    InvalidOpsetImport { domain: String, version: u64 },
}
```

### 3.4 ORT Graph ABI Bridge

```rust
/// Read-only view exposing our IR through ORT's C API.
pub struct OrtGraphView<'a> {
    graph: &'a Graph,
    node_index: OnceCell<Vec<OrtNodeRepr>>,
    name_index: OnceCell<HashMap<&'a str, ValueId>>,
}

/// C-compatible node representation.
#[repr(C)]
pub struct OrtNodeRepr {
    pub index: usize,
    pub op_type: *const c_char,
    pub domain: *const c_char,
    pub input_count: usize,
    pub output_count: usize,
    pub inputs: *const *const c_char,
    pub outputs: *const *const c_char,
}

/// EP's claim over a subgraph.
pub struct SubgraphClaim {
    pub ep_id: EpId,
    pub node_ids: Vec<NodeId>,
    pub input_values: Vec<ValueId>,
    pub output_values: Vec<ValueId>,
    pub meta_def: Option<String>,
}

impl<'a> OrtGraphView<'a> {
    pub fn new(graph: &'a Graph) -> Self;
    pub fn query_capabilities(&self, ep: &dyn ExecutionProvider) -> Vec<SubgraphClaim>;
    pub fn compile_subgraph(&self, ep: &dyn ExecutionProvider, claim: &SubgraphClaim) -> Result<CompiledKernel>;
    pub fn create_exec_context<'b>(&self, kernel: &CompiledKernel, inputs: &'b [&Tensor], outputs: &'b mut [Tensor]) -> EpExecContext<'b>;
}
```

### 3.5 Graph Construction Invariants

After the ONNX loader builds a Graph:
1. Every `ValueId` referenced by a node exists in `graph.values`
2. Every node output `ValueId` is unique (SSA property)
3. Graph inputs and initializers have `producer = None`
4. Symbolic dims with the same protobuf name share the same `SymbolId`
5. Shape inference has been run (best-effort for dynamic shapes)
6. Opset imports match the ops used in the graph

---

## 4. Execution Providers

### 4.1 Unified EP Trait

**All EPs implement the same trait.** No distinction between "native" and "plugin" at the
trait level. The difference is only in how they're loaded (Rust crate vs dlopen).

```rust
/// The core EP interface. Every EP crate implements this.
pub trait ExecutionProvider: Send + Sync {
    /// EP identifier (snake_case, e.g. "cuda_ep", "cpu_ep", "mlx_ep").
    fn name(&self) -> &str;
    fn device_type(&self) -> DeviceType;
    fn device_id(&self) -> DeviceId;

    /// Initialize (allocate device resources, load libraries).
    fn initialize(&mut self, config: &EpConfig) -> Result<()>;
    /// Shutdown (release device resources).
    fn shutdown(&mut self) -> Result<()>;

    /// Can this EP run this op with these shapes/layouts?
    fn supports_op(&self, op: &Node, shapes: &[Shape], layouts: &[TensorLayout]) -> KernelMatch;

    /// Get/create a kernel for this op.
    fn get_kernel(&self, op: &Node, shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>>;

    /// Allocate device memory.
    fn allocate(&self, size: usize, alignment: usize) -> Result<DeviceBuffer>;
    fn deallocate(&self, buffer: DeviceBuffer) -> Result<()>;

    /// Copy between host↔device or device↔device.
    fn copy(&self, src: &DeviceBuffer, dst: &mut DeviceBuffer, size: usize) -> Result<()>;
    /// Async copy (returns fence).
    fn copy_async(&self, src: &DeviceBuffer, dst: &mut DeviceBuffer, size: usize) -> Result<Fence>;

    /// Synchronize all pending operations.
    fn sync(&self) -> Result<()>;

    /// Export this EP as an ORT-compatible C ABI plugin.
    /// Allows this EP to be loaded by upstream ORT.
    fn as_ort_plugin(&self) -> Option<OrtPluginExport> { None }

    /// EP-specific graph optimization passes.
    /// Runs after generic optimizer, before placement.
    /// Use for EP-specific fusions (e.g. QNN subgraph fusion, TRT layer fusion).
    fn custom_passes(&self) -> Vec<Box<dyn OptimizerPass>> { vec![] }

    /// EP can claim specific nodes it wants to handle.
    /// Claimed nodes bypass cost-model placement and go directly to this EP.
    /// Use for EPs that know better than the cost model (e.g. entire subgraph offload).
    fn claim_nodes(&self, graph: &Graph) -> Vec<NodeId> { vec![] }

    /// Save EP-compiled context (TensorRT engines, QNN graphs, etc.).
    /// Serialized into the compilation cache (§41) so subsequent loads
    /// skip EP-side compilation entirely.
    fn save_context(&self) -> Result<Option<EpContext>> { Ok(None) }

    /// Restore EP context from cache. EP can skip its compilation step.
    fn load_context(&mut self, ctx: &EpContext) -> Result<()> { Ok(()) }
}

/// Opaque EP-compiled artifact. Serializable to/from disk.
///
/// This is the **runtime form** of a compiled-EP context. Its **on-disk /
/// interchange form** is the ORT `EPContext` contrib node (domain
/// `com.microsoft`) that embeds this blob directly in an ONNX graph — see
/// §57 for the full node schema, the load/dump paths, and the exact mapping
/// between this struct's `data`/`ep_name`/`ep_version` fields and the node's
/// `ep_cache_context`/`source`/`ep_sdk_version` attributes.
pub struct EpContext {
    /// EP that produced this context.
    pub ep_name: String,
    /// EP version (invalidate cache if EP version changes).
    pub ep_version: String,
    /// Opaque compiled blob (TRT engine, QNN context binary, etc.).
    pub data: Vec<u8>,
    /// Nodes this context covers (for partial graph compilation).
    pub covered_nodes: Vec<NodeId>,
    /// Hardware fingerprint (invalidate if hardware changes).
    pub device_fingerprint: String,
}

/// Macro: generate ORT C ABI export for any EP crate.
/// Makes our EP loadable by upstream ORT as a plugin .so/.dylib.
#[macro_export]
macro_rules! export_ort_plugin {
    ($ep_factory:expr, $register_fn:ident) => {
        #[no_mangle]
        pub extern "C" fn $register_fn(
            options: *mut OrtSessionOptions,
            keys: *const *const std::ffi::c_char,
            values: *const *const std::ffi::c_char,
            num_keys: usize,
        ) -> *mut OrtStatus {
            // Parse config from keys/values
            // Instantiate EP via $ep_factory
            // Register with ORT session options
            // Return null on success, OrtStatus* on error
            todo!()
        }
    };
}
```

### 4.2 Kernel Trait

```rust
/// A kernel ready to execute a specific op with specific shapes.
pub trait Kernel: Send {
    /// Execute. Inputs/outputs are on the correct device.
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()>;

    /// Estimated FLOPs (for cost model).
    fn estimated_flops(&self) -> Option<u64>;

    /// Can this kernel handle non-contiguous (strided) input at index?
    fn supports_strided_input(&self, input_idx: usize) -> bool;

    /// Preferred output layout (kernel writes in this layout most efficiently).
    fn preferred_output_layout(&self) -> Option<TensorLayout>;

    /// Can this kernel be captured in a CUDA graph?
    fn cuda_graph_compatible(&self) -> bool { false }
}

pub enum KernelMatch {
    Supported {
        cost: Cost,
        required_input_layouts: Option<Vec<TensorLayout>>,
        output_layouts: Vec<TensorLayout>,
    },
    Unsupported,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum DeviceType {
    Cpu, Cuda, Rocm, CoreMl, Mlx, WebGpu, Qnn, OpenVino, Custom(u32),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct DeviceId {
    pub device_type: DeviceType,
    pub index: u32,
}
```

### 4.3 Op Registry

```rust
/// Maps (op_type, domain, opset) → kernel factory.
pub struct OpRegistry {
    entries: HashMap<OpKey, Box<dyn KernelFactory>>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct OpKey {
    pub op_type: String,
    pub domain: String,
    pub since_version: u64,
}

pub trait KernelFactory: Send + Sync {
    fn create(&self, node: &Node, input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>>;
}

impl OpRegistry {
    pub fn register(&mut self, key: OpKey, factory: Box<dyn KernelFactory>);
    /// Lookup best match (highest since_version ≤ graph's opset).
    pub fn lookup(&self, op_type: &str, domain: &str, opset: u64) -> Option<&dyn KernelFactory>;
}
```

### 4.4 CPU EP Kernel Example (C++ FFI)

```cpp
// native-eps/cpu/src/kernels/matmul.cpp
// Ported from onnxruntime/core/providers/cpu/math/matmul.cc
// Links against oneDNN for optimized GEMM.

extern "C" {
    int cpu_matmul_execute(
        const float* A, const int64_t* a_shape, int a_ndim, const int64_t* a_strides,
        const float* B, const int64_t* b_shape, int b_ndim, const int64_t* b_strides,
        float* C, const int64_t* c_shape, int c_ndim,
        int trans_a, int trans_b
    );

    int cpu_matmul_supports_strided(int input_idx);
}
```

```rust
// crates/onnx-runtime-ep-cpu/src/kernels/matmul.rs
extern "C" {
    fn cpu_matmul_execute(
        a: *const f32, a_shape: *const i64, a_ndim: c_int, a_strides: *const i64,
        b: *const f32, b_shape: *const i64, b_ndim: c_int, b_strides: *const i64,
        c: *mut f32, c_shape: *const i64, c_ndim: c_int,
        trans_a: c_int, trans_b: c_int,
    ) -> c_int;

    fn cpu_matmul_supports_strided(input_idx: c_int) -> c_int;
}

pub struct CpuMatMulKernel { trans_a: bool, trans_b: bool }

impl Kernel for CpuMatMulKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        let a = &inputs[0]; let b = &inputs[1]; let c = &mut outputs[0];
        let ret = unsafe {
            cpu_matmul_execute(
                a.data_ptr(), a.shape_ptr(), a.ndim() as c_int, a.strides_ptr(),
                b.data_ptr(), b.shape_ptr(), b.ndim() as c_int, b.strides_ptr(),
                c.data_ptr_mut(), c.shape_ptr(), c.ndim() as c_int,
                self.trans_a as c_int, self.trans_b as c_int,
            )
        };
        if ret == 0 { Ok(()) } else { Err(Error::KernelFailed("cpu_matmul".into())) }
    }
    fn supports_strided_input(&self, idx: usize) -> bool {
        unsafe { cpu_matmul_supports_strided(idx as c_int) != 0 }
    }
    fn preferred_output_layout(&self) -> Option<TensorLayout> { None }
    fn estimated_flops(&self) -> Option<u64> { None }
}
```

### 4.5 Legacy ORT Plugin EP Loading

For EPs built against ORT's C API (third-party, precompiled):

```rust
/// Wraps a dlopen'd ORT EP .so and adapts it to our ExecutionProvider trait.
pub struct LegacyOrtEp {
    lib: Library,
    vtable: LegacyEpVTable,
    name: String,
    device_type: DeviceType,
    compiled_kernels: HashMap<u64, CompiledKernel>,
}

pub struct LegacyEpVTable {
    pub get_capability: unsafe extern "C" fn(*const c_void, *const OrtGraphViewer, *mut *mut OrtCapability, *mut usize) -> *mut OrtStatus,
    pub compile: unsafe extern "C" fn(*const c_void, *const *const OrtNode, usize, *mut *mut OrtComputeNode) -> *mut OrtStatus,
    pub compute: unsafe extern "C" fn(*mut c_void, *mut OrtKernelContext) -> *mut OrtStatus,
    pub release: unsafe extern "C" fn(*mut c_void),
}

impl ExecutionProvider for LegacyOrtEp {
    fn name(&self) -> &str { &self.name }
    fn device_type(&self) -> DeviceType { self.device_type }
    // ... adapts legacy C ABI calls to our trait
}

impl LegacyOrtEp {
    /// Load from .so/.dylib path. The EP must export a registration function.
    pub fn load(path: &Path, config: &EpConfig) -> Result<Self>;
}
```

### 4.6 EP Registry

```rust
pub struct EpRegistry {
    eps: Vec<Box<dyn ExecutionProvider>>,
    priority: Vec<EpId>,  // index into eps
}

impl EpRegistry {
    pub fn register(&mut self, ep: Box<dyn ExecutionProvider>) -> EpId;
    pub fn load_legacy(&mut self, path: &Path, config: &EpConfig) -> Result<EpId>;
    pub fn set_priority(&mut self, order: Vec<EpId>);
    /// Get all EPs that can handle a specific op, with costs.
    pub fn candidates_for_op(&self, op: &Node, shapes: &[Shape], layouts: &[TensorLayout]) -> Vec<(EpId, KernelMatch)>;
}
```

---

## 5. Striding and Layout

### 5.1 The Problem

ORT assumes contiguous tensors at EP boundaries → forces memcpy for layout changes.
Most ops can work with strided input. Explicit Transpose ops exist only because
serialization assumes contiguous.

### 5.2 Stride Arithmetic

```rust
/// Compute row-major contiguous strides for a shape.
pub fn compute_contiguous_strides(shape: &[usize]) -> Vec<i64> {
    let n = shape.len();
    let mut strides = vec![1i64; n];
    for i in (0..n.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1] as i64;
    }
    strides
}

/// Check if strides represent contiguous row-major layout.
pub fn is_contiguous(shape: &[usize], strides: &[i64]) -> bool {
    let expected = compute_contiguous_strides(shape);
    strides == &expected[..]
}

/// Compute output shape for numpy-style broadcasting.
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let max_ndim = a.len().max(b.len());
    let mut result = Vec::with_capacity(max_ndim);
    for i in 0..max_ndim {
        let da = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let db = if i < b.len() { b[b.len() - 1 - i] } else { 1 };
        if da == db { result.push(da); }
        else if da == 1 { result.push(db); }
        else if db == 1 { result.push(da); }
        else { return Err(Error::BroadcastIncompatible); }
    }
    result.reverse();
    Ok(result)
}
```

### 5.3 DLPack Tensor Sharing

DLPack is the standard for zero-copy tensor sharing across frameworks (PyTorch, JAX, CuPy, TVM).
Our `Tensor` natively supports DLPack import/export.

```rust
use dlpack_sys::{DLManagedTensor, DLTensor, DLDevice, DLDataType};

impl Tensor {
    /// Export as DLPack managed tensor. Zero-copy — caller gets a view.
    /// The Tensor must outlive the DLManagedTensor (prevented via ref-counted handle).
    pub fn to_dlpack(&self) -> DLManagedTensor {
        DLManagedTensor {
            dl_tensor: DLTensor {
                data: self.data.as_ptr() as *mut c_void,
                device: self.device.to_dl_device(),
                ndim: self.shape.len() as i32,
                dtype: self.dtype.to_dl_dtype(),
                shape: self.shape.as_ptr() as *mut i64,
                strides: self.strides.as_ptr() as *mut i64,
                byte_offset: self.offset as u64,
            },
            manager_ctx: Arc::into_raw(self.handle.clone()) as *mut c_void,
            deleter: Some(prevent_free_fn),
        }
    }

    /// Import from DLPack. Zero-copy — we take ownership of the managed tensor.
    pub fn from_dlpack(managed: DLManagedTensor) -> Self {
        let dl = &managed.dl_tensor;
        let shape: Vec<usize> = unsafe {
            slice::from_raw_parts(dl.shape, dl.ndim as usize)
        }.iter().map(|&x| x as usize).collect();

        let strides = if dl.strides.is_null() {
            TensorLayout::compute_contiguous_strides(&shape)
        } else {
            unsafe { slice::from_raw_parts(dl.strides, dl.ndim as usize) }.to_vec()
        };

        Self {
            data: DevicePtr::from_raw(dl.data),
            dtype: DataType::from_dl_dtype(dl.dtype),
            shape,
            strides,
            device: DeviceId::from_dl_device(dl.device),
            offset: dl.byte_offset as usize,
            _dlpack_handle: Some(managed), // prevent deleter until we're done
        }
    }
}

impl DeviceId {
    pub fn to_dl_device(&self) -> DLDevice {
        DLDevice {
            device_type: match self.device_type {
                DeviceType::Cpu => 1,      // kDLCPU
                DeviceType::Cuda => 2,     // kDLCUDA
                DeviceType::Rocm => 10,    // kDLROCM
                DeviceType::Mlx => 1,      // Metal UMA appears as CPU
                DeviceType::WebGpu => 15,  // kDLWebGPU (proposed)
                _ => 1,
            },
            device_id: self.index as i32,
        }
    }

    pub fn from_dl_device(dl: DLDevice) -> Self {
        let device_type = match dl.device_type {
            1 => DeviceType::Cpu,
            2 => DeviceType::Cuda,
            10 => DeviceType::Rocm,
            _ => DeviceType::Custom(dl.device_type as u32),
        };
        Self { device_type, index: dl.device_id as u32 }
    }
}
```

**Use cases:**
- **EP boundary:** EPs exchange tensors via DLPack instead of custom formats
- **Python zero-copy:** `session.run_from_dlpack(torch_tensor)` — no memcpy
- **Output sharing:** `torch.from_dlpack(output)` — user gets GPU tensor directly
- **Cross-framework pipelines:** JAX preprocessing → our inference → PyTorch postprocessing

**DLPack v1.0 stream semantics:** The `dl_tensor.device` includes stream info for
proper synchronization. When importing a CUDA tensor from PyTorch, we record its
stream and insert a fence before using it on our compute stream.

### 5.4 TensorView / TensorMut (Zero-Copy Views)

```rust
/// Immutable view of a tensor on any device. No ownership of data.
pub struct TensorView<'a> {
    pub data: DevicePtr,           // raw pointer to data on device
    pub dtype: DataType,
    pub shape: &'a [usize],
    pub strides: &'a [i64],       // in elements
    pub device: DeviceId,
    _marker: PhantomData<&'a ()>,
}

impl<'a> TensorView<'a> {
    pub fn is_contiguous(&self) -> bool { is_contiguous(self.shape, self.strides) }
    pub fn numel(&self) -> usize { self.shape.iter().product() }
    pub fn byte_size(&self) -> usize { self.numel() * self.dtype.byte_size() }
    pub fn data_ptr<T>(&self) -> *const T { self.data.as_ptr() as *const T }
}

/// Mutable view.
pub struct TensorMut<'a> {
    pub data: DevicePtrMut,
    pub dtype: DataType,
    pub shape: &'a [usize],
    pub strides: &'a [i64],
    pub device: DeviceId,
    _marker: PhantomData<&'a mut ()>,
}
```

### 5.4 Layout Propagation Pass

```rust
pub struct LayoutPropagation;

impl OptimizationPass for LayoutPropagation {
    fn name(&self) -> &str { "LayoutPropagation" }

    fn run(&self, graph: &mut Graph, ctx: &PassContext) -> Result<()> {
        // For each node in topological order:
        //   1. Query EP kernel's preferred input layouts
        //   2. If input's current layout matches → no action
        //   3. If mismatch, ask cost model:
        //      a) cost(insert MakeContiguous node)
        //      b) cost(kernel running with wrong layout)
        //      → pick cheaper option
        //   4. If kernel has preferred output layout, annotate output value
        //
        // Layout-agnostic ops (Add, Mul, Relu, etc.) propagate input layout to output.
        // This means a chain of elementwise ops after a transposed MatMul
        // all stay transposed — no copy until something requires contiguous.
        todo!()
    }
}
```

**Concrete example — transpose elimination:**
```
Before:
  MatMul(A[M,K], B[K,N]) → C[M,N] (contiguous, strides=[N,1])
  Transpose(C, perm=[1,0]) → D[N,M] (strides=[1,N] → needs copy to make contiguous)
  LayerNorm(D) → E

After layout propagation:
  MatMul(A, B) → C[M,N] with strides=[1,M] (write transposed directly)
  LayerNorm(C) → E  (LayerNorm accepts strided input, no copy)

Savings: eliminated Transpose op + one full-tensor copy.
```

### 5.5 EP Boundary Contiguization

When a legacy plugin EP requires contiguous input:
```rust
/// Inserted by TransferInsertion pass at EP boundaries where needed.
pub struct MakeContiguous;  // Pseudo-op: copies strided tensor to contiguous buffer

// Only inserted when:
// 1. Source tensor is non-contiguous
// 2. Target EP (or kernel) does not support strided input
// The cost of this copy is included in the placement cost model,
// so the optimizer may keep both ops on the same device to avoid it.
```

---

## 6. Cost Model

### 6.1 Role

Single source of truth for all optimization decisions. No hidden heuristics.

### 6.2 Structure

```rust
pub struct CostModel {
    device_profiles: HashMap<DeviceId, DeviceProfile>,
    transfer_matrix: TransferCostMatrix,
    layout_costs: LayoutCostTable,
    /// Calibration data from profiling runs.
    calibration: Option<CalibrationData>,
}

pub struct DeviceProfile {
    pub name: String,
    pub compute_throughput: HashMap<DataType, f64>,  // FLOPS per second
    pub memory_bandwidth: f64,                        // bytes/sec
    pub launch_overhead: Duration,                    // kernel launch latency
    pub op_costs: HashMap<OpSignature, OpCost>,       // measured overrides
}

pub struct TransferCostMatrix {
    entries: HashMap<(DeviceId, DeviceId), TransferProfile>,
}

pub struct TransferProfile {
    pub latency_base: Duration,    // fixed overhead
    pub bandwidth: f64,            // bytes/sec sustained
    pub is_async_capable: bool,
}
```

### 6.3 Concrete Cost Formulas

```rust
impl CostModel {
    /// MatMul cost: 2*M*N*K FLOPs. Memory-bound if arithmetic intensity < device ratio.
    pub fn matmul_cost(&self, m: usize, n: usize, k: usize, dtype: DataType, device: DeviceId) -> Cost {
        let flops = 2 * m * n * k;
        let profile = &self.device_profiles[&device];
        let compute_time = flops as f64 / profile.compute_throughput[&dtype];
        // Roofline: also compute memory time
        let bytes_read = (m * k + k * n) * dtype.byte_size();
        let bytes_written = m * n * dtype.byte_size();
        let memory_time = (bytes_read + bytes_written) as f64 / profile.memory_bandwidth;
        // Actual time is max of compute-bound and memory-bound
        let time_sec = compute_time.max(memory_time);
        Cost { time: Duration::from_secs_f64(time_sec), memory_bytes: bytes_written }
    }

    /// Transfer cost between devices.
    pub fn transfer_cost(&self, bytes: usize, src: DeviceId, dst: DeviceId) -> Cost {
        if src == dst { return Cost::zero(); }
        let profile = &self.transfer_matrix.entries[&(src, dst)];
        let time = profile.latency_base + Duration::from_secs_f64(bytes as f64 / profile.bandwidth);
        Cost { time, memory_bytes: bytes }
    }

    /// Layout transform cost (copying tensor to different stride order).
    pub fn layout_transform_cost(&self, shape: &[usize], dtype: DataType, device: DeviceId) -> Cost {
        let bytes = shape.iter().product::<usize>() * dtype.byte_size();
        // Essentially a memcpy with gather/scatter
        let profile = &self.device_profiles[&device];
        let time = Duration::from_secs_f64(bytes as f64 / (profile.memory_bandwidth * 0.7));
        Cost { time, memory_bytes: bytes }
    }
}

#[derive(Clone, Debug)]
pub struct Cost {
    pub time: Duration,
    pub memory_bytes: usize,
}

impl Cost {
    pub fn zero() -> Self { Self { time: Duration::ZERO, memory_bytes: 0 } }
}
```

### 6.4 Calibration Protocol

```rust
pub struct CalibrationData {
    /// Measured op times: (op_key, device, shapes) → actual duration.
    pub measurements: HashMap<(OpKey, DeviceId, Vec<Vec<usize>>), Duration>,
}

impl CostModel {
    /// Run a calibration pass: execute each op once, measure actual time.
    pub fn calibrate(&mut self, graph: &Graph, session: &InferenceSession,
                     inputs: &[Tensor]) -> Result<CalibrationData>;

    /// Update model with calibration data (Bayesian update of estimates).
    pub fn apply_calibration(&mut self, data: &CalibrationData);

    /// Serialize cost model for caching (avoid re-calibration).
    pub fn save(&self, path: &Path) -> Result<()>;
    pub fn load(path: &Path) -> Result<Self>;
}
```

---

## 7. Graph Partitioning and Device Placement

### 7.1 Problem

ORT: greedy EP claiming. Locally optimal, globally suboptimal (ignores transfer costs).

### 7.2 ILP Formulation

```
Decision variables:
  x[i,d] ∈ {0,1}  — node i assigned to device d

Objective:
  Minimize Σ_i Σ_d  compute_cost(i,d) * x[i,d]
         + Σ_(i,j) Σ_(d1≠d2)  transfer_cost(edge_ij) * x[i,d1] * x[j,d2]

Subject to:
  Σ_d x[i,d] = 1                    ∀ nodes i    (exactly one device)
  x[i,d] = 0  if device d can't run node i       (feasibility)
  
Linearization (since x*x is quadratic):
  Introduce y[i,j,d1,d2] = x[i,d1] * x[j,d2]
  y[i,j,d1,d2] ≤ x[i,d1]
  y[i,j,d1,d2] ≤ x[j,d2]
  y[i,j,d1,d2] ≥ x[i,d1] + x[j,d2] - 1
```

```rust
pub struct PlacementOptimizer {
    cost_model: Arc<CostModel>,
}

impl PlacementOptimizer {
    pub fn optimize(&self, graph: &Graph, registry: &EpRegistry) -> Result<PlacementPlan> {
        let n = graph.num_nodes();
        if n > 10_000 {
            return self.greedy_fallback(graph, registry);
        }
        // Build ILP using `highs` crate (MIT license, pure Rust bindings)
        // 1. Create decision variables x[i,d] for each (node, device) pair
        // 2. Add feasibility constraints
        // 3. Add linearized transfer cost
        // 4. Solve
        // 5. Extract assignment from solution
        todo!()
    }

    fn greedy_fallback(&self, graph: &Graph, registry: &EpRegistry) -> Result<PlacementPlan> {
        // ORT-style: query EPs in priority order, assign claimed subgraphs
        todo!()
    }
}

pub struct PlacementPlan {
    pub assignments: HashMap<NodeId, DeviceId>,
    pub transfer_edges: Vec<TransferEdge>,
    pub total_cost: Cost,
}

pub struct TransferEdge {
    pub value: ValueId,
    pub src_device: DeviceId,
    pub dst_device: DeviceId,
    pub estimated_bytes: usize,
}
```

### 7.3 Concrete Example

```
Graph: Input → [MatMul] → [LayerNorm] → [GELU] → [MatMul2] → Output
Devices: GPU (cuda:0), CPU

Cost matrix (microseconds):
                    GPU     CPU
MatMul (4096²):     50     2000
LayerNorm:          10      100
GELU:                5       50
MatMul2:            50     2000
GPU→CPU transfer:  200       —
CPU→GPU transfer:  200       —

ILP solution: all on GPU. Total = 50+10+5+50 = 115 μs
Greedy (if CPU EP claims LayerNorm first): 50 + 200 + 100 + 200 + 5 + 50 = 605 μs
                                                ^^^transfer overhead^^^

Savings from ILP: 5.3× faster.
```

---

## 8. Memory Planning

### 8.1 Arena Allocator

```rust
pub struct ArenaAllocator {
    /// Pre-allocated device memory region.
    base: DeviceBuffer,
    /// Total size in bytes.
    capacity: usize,
    /// Allocation state: (offset, size, is_free) sorted by offset.
    blocks: Vec<Block>,
}

struct Block {
    offset: usize,
    size: usize,
    is_free: bool,
    value_id: Option<ValueId>,  // which tensor owns this
}

impl ArenaAllocator {
    /// Allocate from the arena. Uses best-fit with size-class bucketing.
    pub fn allocate(&mut self, size: usize, alignment: usize) -> Result<ArenaSlot>;
    /// Free a slot (mark as available for reuse).
    pub fn free(&mut self, slot: ArenaSlot);
    /// Current utilization.
    pub fn utilization(&self) -> f64;
    /// Peak usage so far.
    pub fn peak_usage(&self) -> usize;
}
```

### 8.2 Interference Graph and Buffer Aliasing

```rust
pub struct MemoryPlanner;

impl MemoryPlanner {
    pub fn plan(&self, schedule: &ExecSchedule, graph: &Graph) -> MemoryPlan {
        // 1. Compute lifetime intervals: (first_use, last_use) for each value
        // 2. Build interference graph: edge between values with overlapping lifetimes
        // 3. Graph coloring (greedy, largest-first): assign colors = memory slots
        //    Values with same color share a buffer
        // 4. Size each slot = max(sizes of values in that color class)
        // 5. Compute arena total = sum of all slot sizes (with alignment padding)
        todo!()
    }
}

/// Lifetime of a value in the execution schedule.
pub struct ValueLifetime {
    pub value: ValueId,
    pub first_use: usize,   // schedule step index
    pub last_use: usize,    // schedule step index
    pub size_bytes: usize,  // buffer size needed
    pub device: DeviceId,   // on which device
    pub alignment: usize,
}

pub struct MemoryPlan {
    pub arenas: HashMap<DeviceId, ArenaLayout>,
    pub aliases: Vec<AliasGroup>,
    pub in_place: Vec<InPlaceOp>,
    pub persistent: Vec<PersistentAlloc>,
    pub staging: Vec<StagingBuffer>,
    pub total_scratch_bytes: HashMap<DeviceId, usize>,
}

pub struct AliasGroup {
    pub slot_id: usize,
    pub size: usize,
    pub values: Vec<ValueId>,  // all share this memory slot
}

pub struct InPlaceOp {
    pub node: NodeId,
    pub input_idx: usize,
    pub output_idx: usize,
}
```

### 8.3 Alignment Rules

| Device | Alignment | Reason |
|--------|-----------|--------|
| CPU (AVX-512) | 64 bytes | Vectorized ops |
| CPU (AMX) | 64 bytes | Tile registers |
| CUDA | 256 bytes | Coalesced access + TMA |
| WebGPU | 256 bytes | Buffer alignment spec |
| CoreML | 16 bytes | Metal buffer alignment |

---

## 9. Async Data Transfer

### 9.1 Stream and Fence Types

```rust
/// A compute or transfer stream on a device.
pub enum Stream {
    CudaStream(CudaStreamHandle),
    MetalCommandQueue(MetalQueueHandle),
    HostThread(ThreadId),
    WebGpuQueue(WgpuQueueHandle),
}

/// A synchronization point. Signals when an async operation completes.
pub enum Fence {
    CudaEvent(CudaEventHandle),
    MetalEvent(MetalEventHandle),
    Completed,  // already done (for host-side sync ops)
}

impl Fence {
    /// Block until the fence is signaled.
    pub fn wait(&self);
    /// Check if signaled without blocking.
    pub fn is_ready(&self) -> bool;
}
```

### 9.2 Transfer Scheduler

```rust
pub struct TransferScheduler {
    /// Per-device dedicated transfer streams (separate from compute).
    transfer_streams: HashMap<DeviceId, Stream>,
    /// Pending async transfers.
    pending: Vec<PendingTransfer>,
}

pub struct PendingTransfer {
    pub src: DeviceBuffer,
    pub dst: DeviceBuffer,
    pub fence: Fence,
    pub size_bytes: usize,
    pub needed_by: NodeId,      // compute node that depends on this
    pub issued_after: NodeId,   // compute node that produced the source
}

impl TransferScheduler {
    /// Issue a transfer as soon as source data is ready.
    pub fn schedule(&mut self, transfer: TransferEdge, src_done_fence: &Fence) -> Fence;

    /// Coalesce small transfers into one (if total < threshold and contiguous).
    pub fn coalesce(&mut self, transfers: &[TransferEdge]) -> Vec<TransferEdge>;
}
```

### 9.3 DAG Executor State Machine

```rust
pub enum NodeState {
    /// All dependencies met, ready to launch.
    Ready,
    /// Launched on device stream, waiting for completion.
    Running { fence: Fence },
    /// Waiting for a transfer fence before it can run.
    WaitingFence { fence: Fence },
    /// Completed.
    Done,
}

pub struct DagExecutor {
    schedule: Vec<NodeId>,  // topological order
    states: HashMap<NodeId, NodeState>,
    streams: HashMap<DeviceId, Stream>,
}

impl DagExecutor {
    /// Execute the entire graph asynchronously.
    pub async fn execute(&mut self, graph: &Graph, plan: &PlacementPlan) -> Result<()> {
        // For each node in schedule order:
        //   1. Check if all input fences are ready
        //   2. If transfer needed: issue async transfer, set state to WaitingFence
        //   3. When fence ready: launch kernel on device stream, set state to Running
        //   4. Record output fence, mark Done
        // All operations are non-blocking; use event-driven execution.
        todo!()
    }
}
```

### 9.4 Double Buffering for LLM Decode

```
Step N:
  GPU compute stream:  [Attention + FFN]──produces logits──►
  GPU→CPU transfer:                        [xfer logits]────►
  CPU:                                                       [Sampling]

Step N+1 (overlapped):
  GPU compute stream:  ◄──[Attention + FFN]─────────────────►
  Transfer uses buffer B while step N used buffer A.
  No stall.
```

---

## 10. Compute-Communication Overlap

### 10.1 Motivation

In tensor-parallel or pipeline-parallel settings, All-Reduce/All-Gather communication
can dominate runtime. The key technique (from DeepSeek V3, SGLang, Megatron):
overlap communication of layer N with computation of layer N+1.

### 10.2 Micro-Chunk Overlap

```
Layer N computation:     [chunk1][chunk2][chunk3][chunk4]
Layer N all-reduce:              [AR_c1 ][AR_c2 ][AR_c3 ][AR_c4]
Layer N+1 computation:                   [c1    ][c2    ][c3    ][c4]
                                          ↑ uses AR_c1 result

→ Communication of chunk K overlaps with computation of chunk K+1.
→ Pipeline bubble reduced from full all-reduce latency to one chunk's latency.
```

```rust
pub struct OverlapScheduler {
    /// Number of micro-chunks to split each tensor-parallel operation into.
    pub num_chunks: usize,
    /// Communication stream (separate from compute).
    pub comm_stream: Stream,
}

impl OverlapScheduler {
    /// Split an all-reduce into chunked async operations interleaved with compute.
    pub fn schedule_overlap(
        &self,
        compute_op: &Node,      // layer N+1 compute
        comm_op: &AllReduceOp,  // layer N communication
    ) -> Vec<MicroStep>;
}

pub enum MicroStep {
    Compute { chunk_idx: usize, stream: Stream },
    Communicate { chunk_idx: usize, stream: Stream },
    WaitFence(Fence),
}
```

### 10.3 Async Weight Loading Overlap

For large models loaded layer-by-layer (offloading scenario):

```rust
/// While computing layer N on GPU, prefetch layer N+1 weights from CPU/disk to GPU.
pub struct WeightPrefetcher {
    prefetch_stream: Stream,
    buffer: DoubleBuffer,  // ping-pong between two staging areas
}

impl WeightPrefetcher {
    pub fn prefetch_layer(&mut self, layer_idx: usize, model: &ModelInstance) -> Fence;
}
```

---

## 11. Dynamic Shape Specialization

### 11.1 Shape-Keyed Kernel Cache

```rust
pub struct KernelCache {
    entries: HashMap<KernelCacheKey, CachedKernel>,
    max_entries: usize,
    stats: KernelCacheStats,
}

pub struct KernelCacheKey {
    pub ep_id: EpId,
    pub op_key: OpKey,
    pub shapes: Vec<Vec<usize>>,
}

pub struct CachedKernel {
    pub kernel: Box<dyn Kernel>,
    pub last_used: Instant,
    pub use_count: u64,
}

pub struct KernelCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub recompilations: u64,
}

impl KernelCache {
    pub fn get_or_create(&mut self, key: KernelCacheKey, factory: &dyn KernelFactory, node: &Node) -> Result<&dyn Kernel>;
    pub fn evict_lru(&mut self, count: usize);
    pub fn hit_rate(&self) -> f64 { self.stats.hits as f64 / (self.stats.hits + self.stats.misses) as f64 }
}
```

### 11.2 Shape Bucketing

```rust
/// Round shapes up to reduce cache entries. E.g. seq_len → next power of 2.
pub fn bucket_shapes(shapes: &[Vec<usize>], strategy: &BucketStrategy) -> Vec<Vec<usize>> {
    shapes.iter().map(|shape| {
        shape.iter().enumerate().map(|(dim_idx, &size)| {
            match strategy.rules.get(&dim_idx) {
                Some(BucketRule::PowerOfTwo) => size.next_power_of_two(),
                Some(BucketRule::RoundUp(multiple)) => ((size + multiple - 1) / multiple) * multiple,
                None => size,  // keep exact
            }
        }).collect()
    }).collect()
}

pub enum BucketRule {
    PowerOfTwo,
    RoundUp(usize),  // round to multiple of N
}
```

### 11.3 Warmup Protocol

```rust
impl InferenceSession {
    /// Pre-compile kernels for common shapes at session init.
    /// Avoids first-inference latency spike.
    pub fn warmup(&mut self, shapes: &[WarmupShape]) -> Result<()> {
        for ws in shapes {
            // Create dummy tensors with these shapes
            // Run a single inference (discarding output)
            // This populates the kernel cache
        }
        Ok(())
    }
}

pub struct WarmupShape {
    pub input_name: String,
    pub shape: Vec<usize>,
}
```

---

## 12. Weight Loading and Storage

### 12.1 Memory-Mapped Weights

```rust
pub struct WeightStore {
    mappings: Vec<MmapRegion>,
    index: HashMap<String, WeightRef>,
}

pub struct MmapRegion {
    #[cfg(unix)]    mmap: memmap2::Mmap,
    #[cfg(windows)] mmap: memmap2::Mmap,  // uses MapViewOfFile internally
    path: PathBuf,
    size: usize,
}

pub struct WeightRef {
    pub mapping_idx: usize,
    pub offset: usize,
    pub length: usize,
    pub dtype: DataType,
    pub shape: Vec<usize>,
}

impl WeightStore {
    /// Zero-copy view into mmap'd data.
    pub fn get(&self, name: &str) -> Option<TensorView<'_>>;

    /// Upload specific weight to device (for non-UMA systems).
    /// Returns fence for async upload.
    pub fn upload_to_device(&self, name: &str, device: DeviceId, ep: &dyn ExecutionProvider) -> Result<(DeviceBuffer, Fence)>;
}
```

### 12.2 Format Parsers

```rust
/// Safetensors format: JSON header + raw tensor data.
pub fn load_safetensors(path: &Path) -> Result<WeightStore> {
    // 1. Read first 8 bytes = header_size (u64 LE)
    // 2. mmap the file
    // 3. Parse JSON header (offset 8..8+header_size)
    //    Header maps tensor_name → { dtype, shape, data_offsets: [start, end] }
    // 4. Each tensor's data is at file offset (8 + header_size + data_offset_start)
    // 5. Build WeightRef index pointing into the mmap
    todo!()
}

/// GGUF format: metadata + tensor info + tensor data.
pub fn load_gguf(path: &Path) -> Result<WeightStore> {
    // 1. Parse GGUF header: magic, version, tensor_count, metadata_kv_count
    // 2. Parse metadata key-values (model arch, quantization info, etc.)
    // 3. Parse tensor info array: name, n_dims, dims, type, offset
    // 4. Tensor data starts at alignment boundary after header
    // 5. mmap from data start, build WeightRef with quantization type info
    todo!()
}

/// ONNX external data: tensor files referenced by protobuf.
pub fn load_onnx_external(model_path: &Path, data_dir: &Path) -> Result<WeightStore> {
    // Each initializer in the protobuf has external_data with:
    //   location (filename), offset, length
    // mmap each referenced file, build WeightRef
    todo!()
}
```

### 12.3 Device Upload Strategy

```rust
pub enum UploadStrategy {
    /// Upload all weights to device at session init. Fast inference, high memory.
    Eager,
    /// Upload on first use. Lower memory peak, first-inference latency.
    Lazy,
    /// Keep on host, copy per-inference via pinned staging buffer.
    /// For models that don't fit in device memory (offloading).
    Streaming { prefetch_layers: usize },
}
```

---

## 13. Flash Attention Integration

### 13.1 Approach

We don't write our own FlashAttention kernel. We integrate existing implementations
through the fusion pass:

1. **Pattern match** in optimization pass: detect Q/K/V/Scale/Mask/Softmax/V@ pattern
2. **Replace** with a single `FlashAttention` node
3. **Dispatch** to the appropriate implementation based on device:
   - CUDA: flash-attn library (Tri Dao's FlashAttention-3 for Hopper)
   - CPU: xformers-style chunked attention or naive
   - Metal: MLX's built-in efficient attention

### 13.2 Fusion Pattern

```rust
pub struct AttentionFusionPass;

impl OptimizationPass for AttentionFusionPass {
    fn name(&self) -> &str { "AttentionFusion" }

    fn run(&self, graph: &mut Graph, ctx: &PassContext) -> Result<()> {
        // Pattern: MatMul(Q,K^T) → Scale → [Mask] → Softmax → MatMul(·, V)
        // Also match: multi-head variants (Reshape/Transpose around Q/K/V)
        //
        // Replace with FusedAttention node:
        //   inputs: [Q, K, V, optional_mask]
        //   attributes: { num_heads, head_dim, causal, scale }
        //
        // The FusedAttention op is dispatched to flash-attn at runtime.
        todo!()
    }
}
```

### 13.3 Flash Attention Kernel Binding (CUDA)

```rust
// In onnx-runtime-ep-cuda
pub struct FlashAttentionKernel {
    causal: bool,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
}

impl Kernel for FlashAttentionKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        // Call flash_attn_func via C FFI:
        // flash_attn_with_kvcache(q, k, v, k_cache, v_cache, ...)
        // Supports:
        //   - Variable sequence length (packed/varlen)
        //   - Sliding window attention
        //   - GQA (different num_heads for Q vs KV)
        //   - FP8 KV cache
        //   - Paged attention (block table)
        todo!()
    }

    fn cuda_graph_compatible(&self) -> bool { true }
    fn supports_strided_input(&self, _idx: usize) -> bool { true }
}
```

### 13.4 KV Cache Integration

Flash Attention with paged KV cache (from onnx-genai's KV system):

```rust
/// FlashAttention reads KV directly from paged cache — no copy needed.
pub struct PagedFlashAttention {
    /// Block table: maps logical KV position → physical page.
    pub block_table: &[u32],
    pub page_size: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}
```

---

## 14. CUDA Graph Capture

### 14.1 Motivation

LLM decode step has the same graph structure every token (only input_ids change).
CUDA graph capture records the entire decode step as one graph launch, eliminating:
- All kernel launch overhead (~5-10μs per kernel × ~100 kernels = 0.5-1ms saved)
- CPU-side scheduling overhead

### 14.2 Implementation

```rust
pub struct CudaGraphCapture {
    /// Captured graph (opaque CUDA handle).
    graph: Option<CudaGraphHandle>,
    /// Whether capture is active.
    capturing: bool,
    /// Inputs that change between invocations (their device pointers are updated).
    mutable_inputs: Vec<(String, DeviceBuffer)>,
}

impl CudaGraphCapture {
    /// Begin capture. All subsequent CUDA ops on this stream are recorded.
    pub fn begin_capture(&mut self, stream: &CudaStream) -> Result<()>;

    /// End capture. Returns a replayable graph.
    pub fn end_capture(&mut self, stream: &CudaStream) -> Result<()>;

    /// Replay the captured graph (update input pointers first).
    pub fn replay(&self, stream: &CudaStream, new_inputs: &[(String, DeviceBuffer)]) -> Result<()>;

    /// Check if all kernels in the graph support capture.
    pub fn is_capturable(graph: &Graph, plan: &PlacementPlan, ep: &dyn ExecutionProvider) -> bool {
        // All kernels must return cuda_graph_compatible() == true
        // No host-device synchronization points
        // No dynamic shape changes within the captured region
        todo!()
    }
}
```

### 14.3 Usage in LLM Decode

```rust
impl InferenceSession {
    /// First decode step: capture CUDA graph.
    /// Subsequent steps: replay with updated input_ids + position_ids.
    pub fn decode_step_with_graph(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        if self.cuda_graph.graph.is_none() {
            // First call: do regular execution + capture
            self.cuda_graph.begin_capture(&self.compute_stream)?;
            let output = self.run_decode(input_ids)?;
            self.cuda_graph.end_capture(&self.compute_stream)?;
            Ok(output)
        } else {
            // Subsequent calls: just replay
            self.cuda_graph.replay(&self.compute_stream, &[
                ("input_ids", input_ids.device_buffer()),
            ])?;
            Ok(self.get_output("logits")?)
        }
    }
}
```

---

## 15. CUDA EP Kernel Strategy

> **Status:** DECIDED (2026-07, Roy + coordinator, from a decision-grade research pass with
> dated citations). This section specifies *how* the CUDA EP (`onnx-runtime-ep-cuda`, §56 Phase 2)
> actually talks to the GPU — the layer the rest of the design previously under-specified. No
> hand-wavy "call CUDA"; every op has a named implementation and a named Rust binding path.

### 15.1 Decision Summary

The CUDA EP is a **pure-Rust crate** (no Python runtime dependency in the shipping binary). It
reaches the GPU through **one foundation crate (`cudarc`)** plus **`extern "C"` FFI to custom
`.cu` kernels** compiled by `nvcc` in `build.rs`. The stack, by responsibility:

| Layer | Choice | Role |
|-------|--------|------|
| **Foundation / driver + vendor libs** | **`cudarc`** | Driver, runtime, streams, memory, CUDA-graph capture; typed bindings to cuBLAS/cuBLASLt/cuDNN/NCCL/NVRTC |
| **Standard GEMM** (prefill batched, decode GEMV, FP8/FP16/BF16) | **cuBLASLt** via `cudarc::cublaslt` | Battle-tested, auto-tuned; fused GEMM+Bias+Activation epilogue |
| **Custom fused ops** (norms, RoPE, softmax, residual+norm, elementwise fusion) | **CuTe (CUTLASS 3.x C++)** → `extern "C"` → static lib | Hopper WGMMA/TMA; fusions no vendor lib provides |
| **Attention** | Phase 2a **cuDNN fused SDPA** → Phase 2b **FlashAttention-3** C shim | Clean baseline first, peak-H100 later (§13) |
| **Trivial elementwise** | **NVRTC** via `cudarc::nvrtc` | Runtime PTX compile for simple ops — no build step |
| *Supplemental (optional)* | *ThunderKittens* (header-only Hopper DSL) | For specific Hopper attention/SSM kernels where more ergonomic than raw CuTe |

**cuTile is explicitly deferred to Phase 3** (§15.8) — it has no Hopper/SM90 support and is
Python-only today, both disqualifying for an H100/H200-first, pure-Rust runtime.

Primary hardware target is **Hopper (SM90, H100/H200)**, with an **Ampere (SM80) fallback path**
on every custom kernel via `#if __CUDA_ARCH__ >= 900` arch-gating.

**Model-agnostic HARD RULE (project-wide, applies to every kernel below):** all custom kernels
MUST be **shape-driven, dtype-parameterized (C++ templates), and architecture-gated**. There are
**no hardcoded `num_heads` / `head_dim` / model constants** anywhere in a kernel. Attention
parameters (`causal`, `num_heads`, `head_dim`, `scale`) are **runtime arguments**, never
compile-time constants — exactly as the existing `FlashAttentionKernel::new(causal, num_heads,
head_dim)` binding already models them (§13.3). CuTe C++ templates enforce this naturally
(template parameters over shapes/dtypes); cuBLASLt and cuDNN are already shape-generic.

### 15.2 Foundation: `cudarc`

[`cudarc`](https://github.com/chelsea0x3b/cudarc) is the best-in-class Rust CUDA binding and the
**core dependency of `onnx-runtime-ep-cuda`**. It is the foundation of HuggingFace Candle's GPU
backend, is actively maintained, and covers **CUDA 11.4 → 13.3** (feature-flagged).

- **Driver / runtime:** device, context, stream, event, memory (alloc/copy/pinned) — all of §9
  (async transfer) and §14 (CUDA-graph capture, via `cudarc::driver` stream-capture APIs).
- **Vendor libraries as typed bindings:** cuBLAS, **cuBLASLt**, **cuDNN** (8.x/9.x), NCCL (future
  multi-GPU, §10), cuRAND/cuSolver/cuSparse, and **NVRTC** (`compile_ptx`).
- **Three-level API** per library: `sys` (raw bindgen FFI), `result` (Result-returning), `safe`
  (ergonomic RAII). The EP uses `safe` by default and drops to `result`/`sys` only where a needed
  entry point isn't yet wrapped.
- **Dynamic linking by default** (no build-time CUDA toolkit required to `cargo build` the crate
  metadata); static linking supported.

This replaces any bespoke driver FFI: the EP does **not** hand-roll `cuMemAlloc`/`cuLaunchKernel`
bindings — it consumes `cudarc` and reserves hand-written FFI strictly for our own `extern "C"`
kernel launchers (§15.4) and the FA3 shim (§13, §15.5).

### 15.3 Standard GEMM: cuBLASLt

All standard GEMM paths route to **cuBLASLt** (`cudarc::cublaslt`), preferred over plain cuBLAS
because of its **fused epilogue**:

- `cublasLtMatmul` with `CUBLASLT_MATMUL_DESC_EPILOGUE` fuses **GEMM + Bias + Activation**
  (`CUBLASLT_EPILOGUE_BIAS_GELU`, `..._BIAS_RELU`, `..._BIAS_SILU`) in one launch — natively since
  **CUDA 12.0**. This subsumes the design's earlier "Fused GEMM+Bias+Act" CuTe entry for the
  standard-shape case.
- **FP8** (E4M3/E5M2) GEMM with per-tensor / per-token scaling on Hopper, plus FP16/BF16 — all
  auto-tuned, production-mature (TensorRT-LLM, FasterTransformer).
- **Batched GEMM** for prefill; **GEMV** (M=1) for decode routes here first and is profiled — a
  custom CuTe decode kernel is written **only if** profiling shows cuBLASLt tall-skinny GEMV is a
  bottleneck.

| Op | cuBLASLt path |
|----|---------------|
| GEMM (FP16/BF16/FP8, prefill batched) | `cublasLtMatmul` / batched |
| GEMM + Bias + GELU/SiLU/ReLU (fused) | `CUBLASLT_EPILOGUE_BIAS_*` |
| GEMV (M=1 decode) | `cublasLtMatmul` (profile; custom fallback) |
| FP8 GEMM (H100) | cuBLASLt FP8 path |

### 15.4 Custom Fused Kernels: CuTe (CUTLASS 3.x)

CuTe — CUTLASS 3.x's C++ template/layout abstraction — is the **custom-kernel path**, not the
whole story. It owns the fusions no vendor library provides. CUTLASS is production-mature (in use
since 2017; underpins cuDNN/TensorRT) and gives **first-class Hopper SM90 support**.

CuTe models:
- **Layouts** as algebraic objects (compose, divide, complement)
- **Tiling** as layout transformations
- **Data movement** (shared-memory staging, register tiling, TMA) as layout operations

**Which ops go to CuTe** (vs vendor libs):

| Op | Implementation | Reason |
|----|----------------|--------|
| GEMM (standard shapes) | cuBLASLt (§15.3) | Battle-tested, auto-tuned |
| GEMM+Bias+Act (standard) | cuBLASLt epilogue (§15.3) | Vendor-fused since CUDA 12.0 |
| LayerNorm / RMSNorm | CuTe + warp reduction, one template per dtype | Not in cuBLAS |
| Residual + LayerNorm (fused) | CuTe | Cross-op fusion, not in cuDNN |
| RoPE | Custom CUDA C (position-indexed elementwise) | Simple; no CUTLASS needed |
| Softmax | Custom CUDA C (online / safe-softmax) | Not in cuBLAS |
| Quantized MatMul (INT4×FP16) | CuTe (dequant+GEMM fusion) | Custom fusion |
| Attention | cuDNN / FA3 (§13, §15.5) | Specialized — not hand-rolled in CuTe |

**Rust consumption pattern** (the design's chosen path — `native-eps/cuda/` C++ kernels, §56):

1. **Instantiate** the CUTLASS kernel variant in a `.cu` file with **template** type/shape params.
2. **Export** an `extern "C"` launcher, e.g. `void launch_fused_residual_layernorm(...)`.
3. **Compile** via `nvcc` in `build.rs` with `-gencode arch=compute_90,code=sm_90` (+ SM89/SM80).
4. **Archive** into a static `libcuda_kernels.a`; `cargo:rustc-link-lib=static=cuda_kernels`.
5. **Declare** matching `extern "C"` signatures in Rust and call them from the EP kernels.

```cpp
// native-eps/cuda/src/fused_residual_layernorm.cu
#include <cute/tensor.hpp>

// dtype-parameterized (T), shape-driven (hidden_size is a runtime arg) — no model constants.
template<typename T, int kBlockSize = 1024>
__global__ void fused_residual_layernorm(
    T const* residual,   // [batch, seq, hidden]
    T const* input,      // [batch, seq, hidden]
    T const* gamma,      // [hidden]
    T const* beta,       // [hidden]
    T*       output,     // [batch, seq, hidden]
    int      hidden_size,
    float    eps
) {
    using namespace cute;
    int idx = blockIdx.x;                                   // one (batch, seq) row per block
    auto layout = make_layout(make_shape(hidden_size));     // CuTe layout over the hidden dim
    // fused residual add → mean/var in registers → normalize → gamma/beta → write.
    // No intermediate buffer for the residual add.
}

// extern "C" launcher (the Rust FFI boundary). Dtype is dispatched at runtime.
extern "C" void launch_fused_residual_layernorm(
    const void* residual, const void* input,
    const void* gamma, const void* beta, void* output,
    int batch_seq, int hidden_size, float eps, int dtype, cudaStream_t stream);
```

#### Hopper-specific: TMA + WGMMA (with SM80 fallback)

On SM90 (H100/H200), CuTe gives direct access to **TMA** (Tensor Memory Accelerator — async
global→shared copy without occupying warps) and **WGMMA** (Warpgroup MMA — async warpgroup GEMM).
Every such kernel is **arch-gated** so it still runs on Ampere:

```cpp
#if __CUDA_ARCH__ >= 900
    // Hopper path: TMA async copy + WGMMA
    auto tma_load = make_tma_copy(SM90_TMA_LOAD{}, source_tensor, smem_layout);
    cute::copy(tma_load, source_tensor, shared_tensor);   // global → shared, frees warps
#else
    // Ampere (SM80) fallback: cp.async + mma.sync
#endif
```

### 15.5 Attention (phased) — reconciles with §13

Attention is **not** hand-written in CuTe. It follows the phasing already introduced in §13, now
pinned to concrete Rust bindings:

- **Phase 2a — cuDNN fused SDPA** via `cudarc::cudnn`. cuDNN 9.x exposes fused Scaled-Dot-Product
  Attention through a clean **C API** (GQA, causal mask, sliding window, FP8 on H100). This is the
  **fast-to-integrate baseline** — good enough throughput, no custom build step. (cuDNN wrapping in
  `cudarc` is less mature than cuBLAS; the EP drops to the `result`/`sys` layer where needed.)
  Limitation: no paged-KV out of the box.
- **Phase 2b — FlashAttention-3** (Tri Dao, Hopper). Peak H100 attention (~1.5–2× FA2), FP8,
  GQA, **paged-KV**. FA3 ships no official C ABI or Rust crate, so we vendor its `hopper/` csrc and
  add a **hand-written `flash_attn_shim.cu`** `extern "C"` wrapper (excluding the PyTorch-dependent
  files), compiled as part of `native-eps/cuda/`. This is the **biggest build-complexity item in
  the stack** — a one-time investment. This is the runtime behind the existing §13.3
  `FlashAttentionKernel` binding and the §13.4 `PagedFlashAttention` path; FA3 is **Hopper-only**,
  so non-Hopper deployments fall back to the Phase 2a cuDNN SDPA path.

See §13 for the fusion pass, the `FlashAttentionKernel` binding, and paged-KV integration — this
section only fixes *which library* fulfills them. `FlashAttentionKernel::new(causal, num_heads,
head_dim)` already takes these as **runtime** arguments and satisfies the §15.1 hard rule.

### 15.6 Trivial Elementwise: NVRTC

Simple standalone elementwise ops (GELU/SiLU standalone, add/mul/scale, casts) can skip the build
step entirely: `cudarc::nvrtc::compile_ptx(src)` compiles a CUDA-C source string to PTX **at
runtime** and launches it. Used for ops too trivial to warrant a `.cu` file in the static lib;
fused elementwise variants still go to a `.cu`/CuTe kernel.

### 15.7 "If We Start Phase 2 Tomorrow" — Crate Layout & Build Sketch

```
onnx-runtime-ep-cuda/
├── build.rs                  # nvcc: native-eps/cuda/*.cu → libcuda_kernels.a
├── Cargo.toml                # cudarc = { features = ["cuda-12090","cublas","cublaslt","cudnn","nccl"] }
└── src/
    ├── provider.rs           # CudaExecutionProvider (cudarc::driver device/context)
    ├── allocator.rs          # device allocator (cudarc)
    ├── stream.rs             # stream + CUDA-graph capture (cudarc)
    └── kernels/
        ├── gemm.rs           # cuBLASLt via cudarc::cublaslt::safe
        ├── attention.rs      # Phase 2a cuDNN SDPA; Phase 2b flash_attn_shim FFI
        ├── fused.rs          # extern "C" FFI → libcuda_kernels.a (CuTe kernels)
        └── elementwise.rs    # NVRTC via cudarc::nvrtc for trivial ops

native-eps/cuda/
├── include/kernels.h         # extern "C" launcher declarations
└── src/
    ├── fused_residual_layernorm.cu   # CuTe (SM90 TMA path + SM80 fallback)
    ├── rms_norm.cu                   # CuTe, dtype-templated
    ├── rope.cu                       # custom CUDA C (elementwise, position-indexed)
    ├── softmax.cu                    # custom CUDA C (online/safe softmax)
    ├── gelu_silu.cu                  # standalone elementwise (or NVRTC)
    └── flash_attn_shim.cu            # Phase 2b: extern "C" shim around FA3 hopper/ csrc
```

```rust
// build.rs sketch — nvcc-compile each .cu with CUTLASS headers, multi-arch, archive.
fn main() {
    let nvcc = std::env::var("NVCC").unwrap_or_else(|_| "nvcc".into());
    for src in glob("native-eps/cuda/src/*.cu") {
        Command::new(&nvcc).args([
            "-O3", "-std=c++17",
            "-gencode", "arch=compute_90,code=sm_90",   // Hopper H100/H200 (primary)
            "-gencode", "arch=compute_89,code=sm_89",   // Ada L40
            "-gencode", "arch=compute_80,code=sm_80",   // Ampere A100 (fallback)
            "--include-path", "third_party/cutlass/include",
            "-c", &src, "-o", &obj,
        ]).status().unwrap();
    }
    // archive objs → libcuda_kernels.a
    println!("cargo:rustc-link-lib=static=cuda_kernels");
    println!("cargo:rustc-link-lib=dylib=cudart");
}
```

Build-complexity note: CUTLASS template instantiation compiles slowly (minutes/kernel) — mitigate
with `sccache`/`ccache`. `cudarc` needs no C++ build for vendor libs. The FA3 shim (§15.5) is the
one heavyweight, one-time integration.

### 15.8 cuTile — Deferred to Phase 3

Justin asked whether the Phase-2 CUDA EP should use NVIDIA **cuTile** "or a better choice." The
answer: **not cuTile for Phase 2.** NVIDIA CUDA Tile launched with **CUDA 13.1** (a tile-first
Python DSL over a new "CUDA Tile IR", philosophically closer to Triton than to CUTLASS CuTe). Two
disqualifiers for this project:

1. **No Hopper / SM90 support.** The CUDA 13.1 blog states CUDA Tile "is supported on NVIDIA
   Ampere, NVIDIA Ada and NVIDIA Blackwell (compute capability 8.x, 10.x, 11.x and 12.x) products
   only"; the `tileiras` compiler (v13.2) "only supports Blackwell GPU and Ampere/Ada GPU. Hopper
   GPU will be supported in the coming versions"
   ([NVIDIA CUDA 13.1 blog, 2025](https://developer.nvidia.com/blog/nvidia-cuda-13-1-powers-next-gen-gpu-programming-with-nvidia-cuda-tile-and-performance-gains/);
   [NVIDIA/cutile-python README, 2025](https://github.com/nvidia/cutile-python)). Hopper = SM90 =
   CC 9.x = **our primary H100/H200 target** — directly excluded.
2. **Python-only; no C++ or Rust path.** cuTile Python is the sole user-facing interface; the blog
   notes a C++ implementation is only "planned for a future CUDA release," with no Rust path and a
   Python + `tileiras` runtime dependency — **incompatible with a pure-Rust shipping binary.**

**Re-evaluate cuTile in Phase 3** when (a) Hopper support ships, (b) a C++ / standalone-AOT path
lands, and (c) production adoption appears — earliest realistic window ~CUDA 14.x / 2027.

#### Also watched for Phase 3 (not adopted for Phase 2)

- **CuTe DSL (CUTLASS 4.x Python).** CUTLASS 4.6.1 (July 2026) adds a Python DSL and an
  **experimental `cute.compile_to` AOT path** to "build customized compile-execute pipelines
  outside of Python" ([CUTLASS README, 2026](https://github.com/NVIDIA/cutlass)); FlashAttention-4
  is authored in it. The DSL graduates beta "by end of summer 2026." This *could* one day let
  CuTe-DSL kernels be AOT-compiled and linked into a Rust runtime — **watch for Phase 3.** For
  Phase 2 it is ignored: **CUTLASS 3.x C++ templates are sufficient and production-proven.**

#### Explicitly rejected for production (Phase 2)

- **Rust-CUDA / `rustc_codegen_nvvm`** — revived but self-described "early development" (bugs,
  breaking changes), requires Rust **nightly**, no CUTLASS/WGMMA/TMA integration. Not for a
  shipping runtime.
- **Triton AOT (NVIDIA)** — the AOT path is experimental/fragile and drags a Python
  compile→`.so`→dlopen dependency. Skip.
- **TileLang / Mojo** — Python-only authoring, no Rust/C++ AOT path. Out of scope.

---

## 16. Auto-Tuning Agent Interface

### 16.1 Purpose

An LLM agent (or automated script) can drive performance optimization in a closed loop:
profile → analyze → suggest → apply → verify.

### 16.2 API

```rust
pub struct AutoTuner {
    session: InferenceSession,
    profiler: Profiler,
    cost_model: CostModel,
    history: Vec<TuningStep>,
}

impl AutoTuner {
    /// Profile with given inputs, return structured report.
    pub fn profile(&mut self, inputs: &[Tensor], num_runs: usize) -> Result<ProfilingReport>;

    /// Identify bottlenecks from profiling data.
    pub fn identify_bottlenecks(&self, report: &ProfilingReport) -> Vec<Bottleneck>;

    /// Get optimization suggestions.
    pub fn suggest(&self, report: &ProfilingReport) -> Vec<Suggestion>;

    /// Apply a suggestion. Returns rollback handle.
    pub fn apply(&mut self, suggestion: &Suggestion) -> Result<RollbackHandle>;

    /// Compare two runs.
    pub fn compare(before: &ProfilingReport, after: &ProfilingReport) -> Comparison;

    /// Rollback.
    pub fn rollback(&mut self, handle: RollbackHandle) -> Result<()>;

    /// Full auto-tune loop: try all suggestions, keep improvements.
    pub fn auto_tune(&mut self, inputs: &[Tensor], max_iterations: usize) -> Result<TuningResult>;
}

pub enum Suggestion {
    ChangePlacement { nodes: Vec<NodeId>, target_device: DeviceId },
    FuseOps { nodes: Vec<NodeId>, fusion_type: FusionType },
    ChangeKernel { node: NodeId, kernel_variant: String },
    EnableCudaGraph { region: Vec<NodeId> },
    AdjustChunkSize { op: NodeId, chunk_size: usize },
    EnableOverlap { compute: NodeId, communication: NodeId },
    QuantizeKvCache { dtype: DataType },
    ChangeBatchSize(usize),
}

pub struct Bottleneck {
    pub node: NodeId,
    pub bottleneck_type: BottleneckType,
    pub time_fraction: f64,  // fraction of total time spent here
    pub suggestion: Option<Suggestion>,
}

pub enum BottleneckType {
    ComputeBound,
    MemoryBound,
    TransferBound,
    LaunchOverhead,
    Synchronization,
}

pub struct ProfilingReport {
    pub total_time: Duration,
    pub per_node: HashMap<NodeId, NodeProfile>,
    pub transfers: Vec<TransferProfile>,
    pub memory_peak: usize,
    pub gpu_utilization: f64,
    pub memory_bandwidth_utilization: f64,
}
```

### 16.3 Agent Workflow (for LLM-based optimization)

```
1. Agent calls: tuner.profile(sample_inputs, 10)
2. Agent reads ProfilingReport JSON:
   - "MatMul_0 takes 45% of time, memory-bound (compute utilization 30%)"
3. Agent reasons: "This MatMul is memory-bound → try fusing with subsequent Bias+Relu"
4. Agent calls: tuner.apply(FuseOps { nodes: [matmul_0, bias_0, relu_0], .. })
5. Agent calls: tuner.profile(sample_inputs, 10)  // re-measure
6. Agent compares: 15% speedup? Accept. Regression? Rollback.
```

---

## 17. Debugging and Profiling

### 17.1 Integration with `tracing` Crate

```rust
use tracing::{span, Level, instrument};

#[instrument(skip(inputs, outputs), fields(op = %node.op_type, device = ?device))]
pub fn execute_node(node: &Node, device: DeviceId, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
    // tracing automatically records entry/exit timing
    // Custom fields added:
    tracing::info!(flops = kernel.estimated_flops(), "kernel launched");
    kernel.execute(inputs, outputs)
}
```

### 17.2 Chrome Trace Export

```json
[
  {"name": "MatMul_0", "cat": "compute", "ph": "X", "ts": 1000, "dur": 50,
   "pid": 0, "tid": 0, "args": {"device": "cuda:0", "shapes": [[1,4096,4096]]}},
  {"name": "GPU→CPU transfer", "cat": "transfer", "ph": "X", "ts": 1020, "dur": 200,
   "pid": 0, "tid": 1, "args": {"bytes": 16384, "src": "cuda:0", "dst": "cpu"}},
  {"name": "LayerNorm_0", "cat": "compute", "ph": "X", "ts": 1050, "dur": 10,
   "pid": 0, "tid": 0, "args": {"device": "cuda:0"}}
]
```

### 17.3 Deterministic Replay

```rust
pub struct ReplayCapture {
    pub model_path: String,
    pub inputs: Vec<NamedTensor>,         // captured input tensors
    pub random_seeds: HashMap<String, u64>, // for dropout, sampling
    pub placement: PlacementPlan,
    pub env: RuntimeEnv,                  // thread count, device config
}

impl ReplayCapture {
    pub fn save(&self, path: &Path) -> Result<()>;  // binary format
    pub fn load(path: &Path) -> Result<Self>;
    /// Replay: should produce bit-identical output.
    pub fn replay(&self) -> Result<Vec<NamedTensor>>;
}
```

### 17.4 CLI Commands

```bash
# Profile a model
nxrt profile model.onnx --inputs input.npz --runs 100 --output trace.json

# Compare two runs
nxrt compare trace_before.json trace_after.json

# Dump graph at each optimization stage
nxrt dump-passes model.onnx --format dot --output-dir passes/

# Memory analysis
nxrt memory model.onnx --inputs input.npz --output memory_report.json

# Validate (check output matches ORT)
nxrt validate model.onnx --inputs input.npz --reference-output ort_output.npz --tolerance 1e-5
```

---

## 18. Optimization Passes

### 18.1 Pass Pipeline

```rust
pub trait OptimizationPass: Send + Sync {
    fn name(&self) -> &str;
    fn run(&self, graph: &mut Graph, ctx: &PassContext) -> Result<()>;
    /// Invariants that must hold after this pass (checked in debug builds).
    fn postconditions(&self, graph: &Graph) -> Result<()> { graph.validate().map(|_| ()) }
}

pub struct PassContext {
    pub cost_model: Arc<CostModel>,
    pub ep_registry: Arc<EpRegistry>,
    pub target_devices: Vec<DeviceId>,
}

/// No optimization levels — we always run the full pass pipeline.
pub fn default_passes() -> Vec<Box<dyn OptimizationPass>> {
    vec![
        // Graph normalization
        Box::new(ConstantFolding),
        Box::new(ShapeInference),
        Box::new(DeadNodeElimination),
        // Fusion
        Box::new(OpFusion),
        Box::new(AttentionFusionPass),
        // Layout and placement
        Box::new(LayoutPropagation),
        Box::new(PlacementOptimizer::new()),
        Box::new(TransferInsertion),
        // Memory
        Box::new(InPlaceDetection),
        Box::new(MemoryPlanning),
        // Performance
        Box::new(CudaGraphRegionDetection),
        Box::new(OverlapScheduling),
    ]
}
```

### 18.2 Fusion Pattern Matching

```rust
pub struct OpFusion;

impl OptimizationPass for OpFusion {
    fn name(&self) -> &str { "OpFusion" }

    fn run(&self, graph: &mut Graph, ctx: &PassContext) -> Result<()> {
        let patterns = vec![
            FusionPattern::new("MatMul+Bias+Relu", &["MatMul", "Add", "Relu"], "FusedGemm"),
            FusionPattern::new("MatMul+Bias", &["MatMul", "Add"], "FusedMatMulBias"),
            FusionPattern::new("LayerNorm", &["ReduceMean", "Sub", "Pow", "ReduceMean", "Add", "Sqrt", "Div", "Mul", "Add"], "LayerNormalization"),
            FusionPattern::new("Residual+LayerNorm", &["Add", "LayerNormalization"], "FusedResidualLayerNorm"),
            FusionPattern::new("GELU", &["Mul", "Pow", "Mul", "Add", "Mul", "Tanh", "Add", "Mul"], "Gelu"),
        ];

        for pattern in &patterns {
            while let Some(match_) = pattern.find_match(graph) {
                pattern.apply_fusion(graph, &match_)?;
            }
        }
        Ok(())
    }
}

pub struct FusionPattern {
    name: String,
    /// Op sequence to match (must form a connected subgraph).
    ops: Vec<String>,
    /// Replacement op type.
    replacement: String,
}

impl FusionPattern {
    /// Find the next match in the graph (DFS from each node).
    pub fn find_match(&self, graph: &Graph) -> Option<PatternMatch>;
    /// Apply the fusion: remove matched nodes, insert replacement.
    pub fn apply_fusion(&self, graph: &mut Graph, match_: &PatternMatch) -> Result<()>;
}
```

---

## 19. ONNX Loader

### 19.1 Protobuf Parsing

```rust
// Generated by prost from onnx.proto3
pub mod onnx_proto {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

pub fn load_model(path: &Path) -> Result<Graph> {
    // 1. Read and parse protobuf
    let bytes = std::fs::read(path)?;
    let model_proto = onnx_proto::ModelProto::decode(&bytes[..])?;

    // 2. Validate opset imports
    let opset_imports = parse_opset_imports(&model_proto.opset_import);

    // 3. Build Graph from GraphProto
    let graph = build_graph(&model_proto.graph.unwrap(), &opset_imports)?;

    // 4. Load weights (either inline or external data)
    let weights = load_weights(&model_proto, path.parent().unwrap())?;

    // 5. Run shape inference
    let graph = run_shape_inference(graph)?;

    Ok(graph)
}
```

### 19.2 External Data Resolution

```rust
/// Resolve external data references from ONNX model.
pub fn load_weights(model: &ModelProto, model_dir: &Path) -> Result<WeightStore> {
    let mut store = WeightStore::new();

    for initializer in &model.graph.as_ref().unwrap().initializer {
        if initializer.data_location == DataLocation::External as i32 {
            // Parse external_data fields: location, offset, length
            let location = get_external_field(initializer, "location")?;
            let offset: usize = get_external_field(initializer, "offset")?.parse()?;
            let length: usize = get_external_field(initializer, "length")?.parse()?;
            let path = model_dir.join(location);
            store.add_external(&initializer.name, &path, offset, length,
                              parse_dtype(initializer.data_type), &initializer.dims)?;
        } else {
            // Inline data in protobuf
            store.add_inline(&initializer.name, &initializer.raw_data,
                            parse_dtype(initializer.data_type), &initializer.dims)?;
        }
    }
    Ok(store)
}
```

### 19.3 Shape Inference

```rust
pub fn run_shape_inference(mut graph: Graph) -> Result<Graph> {
    let topo = graph.topological_order()?;
    for node_id in topo {
        let node = graph.node(node_id);
        let input_shapes: Vec<Shape> = node.inputs.iter()
            .filter_map(|v| v.map(|id| graph.value(id).shape.clone()))
            .collect();

        // Dispatch to per-op shape inference
        let output_shapes = infer_shapes(&node.op_type, &node.domain,
                                         &input_shapes, &node.attributes)?;

        for (out_id, shape) in node.outputs.iter().zip(output_shapes) {
            graph.value_mut(*out_id).shape = shape;
        }
    }
    Ok(graph)
}
```

---

## 20. Session API

### 20.1 Design Philosophy

**Zero-config by default.** The user should never need to know what an "Execution Provider" is.
The runtime auto-detects available hardware and picks the best execution strategy.

**No IoBinding.** Buffer reuse is an internal optimization, not a user-facing API.
The session automatically reuses output buffers when shapes don't change, and captures
CUDA graphs when it detects stable shapes. Users who need explicit buffer control
pass pre-allocated tensors via DLPack.

Comparison with ORT:
```python
# ORT (verbose, implementation-leaking):
opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
opts.intra_op_num_threads = 4
session = ort.InferenceSession("model.onnx", opts, providers=["CUDAExecutionProvider"])
output = session.run(None, {"input_ids": data})

# Ours (intent-based, zero-config):
session = nxrt.load("model.onnx")
output = session.run(input_ids=data)
```

### 20.2 Core API

```rust
/// Load a model. Auto-detects best available hardware.
/// This is the primary entry point. No configuration needed.
pub fn load(path: impl AsRef<Path>) -> Result<InferenceSession> {
    InferenceSession::builder()
        .model(path)
        .build()
}

impl InferenceSession {
    /// Primary entry point.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Self::builder().model(path).build()
    }

    /// Load from bytes.
    pub fn load_bytes(bytes: &[u8]) -> Result<Self> {
        Self::builder().model_bytes(bytes).build()
    }

    /// Builder for advanced configuration.
    pub fn builder() -> SessionBuilder {
        SessionBuilder::new()
    }

    /// Run inference. Inputs by name.
    pub fn run(&mut self, inputs: &[(&str, &Tensor)]) -> Result<Vec<Tensor>>;

    /// Run from DLPack tensors (zero-copy from PyTorch/JAX).
    /// Also supports pre-allocated output tensors for manual buffer control.
    pub fn run_from_dlpack(&mut self, inputs: &[(&str, DLManagedTensor)]) -> Result<Vec<Tensor>>;

    /// Input/output metadata.
    pub fn inputs(&self) -> &[IoMeta];
    pub fn outputs(&self) -> &[IoMeta];
}

pub struct IoMeta {
    pub name: String,
    pub dtype: DataType,
    pub shape: Shape,
}
```

### 20.3 Internal Buffer Reuse (replaces IoBinding)

IoBinding is an **implementation detail**, not a user API. The session manages it internally:

```rust
/// Internal: tracks output buffers for reuse across runs.
struct OutputBufferCache {
    /// Last-used output buffers, keyed by output name.
    buffers: HashMap<String, DeviceBuffer>,
    /// Shape of last run (for reuse check).
    last_shapes: HashMap<String, Vec<usize>>,
    /// Number of consecutive runs with same shapes (for CUDA graph trigger).
    stable_shape_count: u32,
}

impl OutputBufferCache {
    /// If shape matches last run, reuse the buffer. Otherwise reallocate.
    fn get_or_allocate(&mut self, name: &str, shape: &[usize], dtype: DataType, device: &dyn ExecutionProvider) -> Result<&mut DeviceBuffer>;

    /// After N consecutive stable-shape runs, trigger CUDA graph capture.
    fn should_capture_cuda_graph(&self) -> bool {
        self.stable_shape_count >= 3
    }
}
```

User sees none of this. They just call `session.run()` and it's fast.

### 20.4 Execution Hints

Users can provide placement/memory/scheduling hints to bias the optimizer.
Hints are loaded from three sources (in priority order, later overrides earlier):

1. **Embedded in `inference_metadata.yaml`** (model author distributes with model)
2. **Standalone `execution_hints.json`** (user places next to model file)
3. **Programmatic via builder API** (highest priority)

Schema: [`schema/execution_hints.schema.json`](../schema/execution_hints.schema.json)

**File discovery:**
```
model.onnx
inference_metadata.yaml          # may contain "execution_hints" key
execution_hints.json             # standalone override (optional)
```

The loader checks for `execution_hints.json` in the same directory as the model.
If both exist, they are merged (standalone file wins on conflicts).

**Example `execution_hints.json`:**
```json
{
  "placement": [
    {
      "selector": { "pattern": "layers.*.attention.*" },
      "device": { "type": "gpu" },
      "strength": "force",
      "reason": "Attention must colocate with KV cache on GPU"
    },
    {
      "selector": { "pattern": "layers.0.*", "op_types": ["Embedding"] },
      "device": { "type": "cpu" },
      "strength": "prefer",
      "reason": "Embedding is memory-bound, CPU has more bandwidth"
    },
    {
      "selector": { "layer_range": { "start": 24, "end": 31 } },
      "device": { "type": "cpu" },
      "strength": "prefer",
      "reason": "Last 8 layers offloadable when GPU memory is tight"
    }
  ],
  "memory": [
    {
      "action": "pin",
      "selector": { "pattern": "kv_cache*" },
      "reason": "KV cache must never be evicted"
    },
    {
      "action": "low_priority",
      "selector": { "pattern": "layers.*.mlp.gate_proj.weight" },
      "reason": "MoE inactive expert weights can be evicted first"
    },
    {
      "action": "arena_size",
      "device": { "type": "gpu", "index": 0 },
      "bytes": 4294967296,
      "reason": "Reserve 4GB scratch arena on GPU 0"
    }
  ],
  "scheduling": [
    {
      "action": "cuda_graph_region",
      "selector": { "pattern": "layers.*" },
      "reason": "Entire transformer stack is CUDA-graph-capturable"
    },
    {
      "action": "overlap",
      "selector": { "op_types": ["AllReduce"] },
      "chunk_size": 4,
      "reason": "Overlap all-reduce with next layer compute"
    }
  ]
}
```

**Example embedded in `inference_metadata.yaml`:**
```yaml
# ... other metadata ...
execution_hints:
  placement:
    - selector: { pattern: "layers.*.attention.*" }
      device: { type: gpu }
      strength: force
  memory:
    - action: pin
      selector: { pattern: "kv_cache*" }
```

**Programmatic (Rust):**
```rust
use onnx_runtime_session::{PlacementHint, NodeSelector, DeviceTarget, HintStrength};

let session = InferenceSession::builder()
    .model("model.onnx")
    .placement_hint(PlacementHint {
        selector: NodeSelector::pattern("layers.*.attention.*"),
        device: DeviceTarget::gpu(0),
        strength: HintStrength::Force,
    })
    .memory_hint(MemoryHint::pin("kv_cache*"))
    .build()?;
```

**Programmatic (Python):**
```python
session = nxrt.load("model.onnx", hints={
    "placement": [
        {"selector": {"pattern": "layers.*.attention.*"}, "device": {"type": "gpu"}, "strength": "force"},
    ],
    "memory": [
        {"action": "pin", "selector": {"pattern": "kv_cache*"}},
    ],
})
```

**How hints affect the optimizer:**

| Strength | Effect on ILP cost model |
|----------|-------------------------|
| `prefer` | 10× cost penalty for violating the hint |
| `force` | Hard constraint in ILP (infinite cost / constraint equation) |

The optimizer always finds a valid plan — `prefer` hints can be overridden
if the total cost would be absurd. `force` hints are never violated
(build fails with an error if the forced placement is infeasible).

### 20.4 Device Selection (Intent-Based)

```rust
/// What the user cares about.
#[derive(Default)]
pub enum DevicePreference {
    /// Auto-detect best available. Priority: CUDA > MLX > ROCm > CPU.
    #[default]
    Auto,
    /// Prefer GPU (any kind).
    Gpu,
    /// Specific GPU index (multi-GPU).
    GpuIndex(u32),
    /// Force CPU.
    Cpu,
    /// Explicit device (escape hatch).
    Specific(DeviceId),
}
```

### 20.5 Session Options (Three Tiers)

**Tier 1: Zero-config (99% of users)**
```rust
let session = InferenceSession::load("model.onnx")?;
```

**Tier 2: Common needs (fluent API)**
```rust
let session = InferenceSession::builder()
    .model("model.onnx")
    .device(DevicePreference::Gpu)
    .memory_limit(4 * GB)
    .profiling(true)
    .build()?;
```

**Tier 3: Namespaced key-value options (power users / escape hatch)**
```rust
let session = InferenceSession::builder()
    .model("model.onnx")
    .option("cuda.use_tf32", "true")
    .option("cuda.device_id", "1")
    .option("cpu.threads", "8")
    .option("memory.arena_size", "2G")
    .option("memory.weight_upload", "lazy")
    .option("profiler.output", "/tmp/trace.json")
    .option("custom_ops.library", "/path/to/custom_ops.so")
    .build()?;
```

All options are namespaced with dot notation:

| Namespace | Keys | Default | Notes |
|-----------|------|---------|-------|
| `cuda.*` | `use_tf32`, `device_id`, `arena_mb` | tf32=true, device=0 | CUDA-specific tuning |
| `cpu.*` | `threads`, `pin_memory` | threads=physical_cores | CPU EP config |
| `memory.*` | `arena_size`, `weight_upload` | auto, eager | Memory management |
| `profiler.*` | `output`, `mode` | off | Profiling config |
| `custom_ops.*` | `library` | none | Custom op registration |
| `scheduler.*` | `cuda_graph`, `overlap` | auto, auto | Execution strategy |

**What we delete from ORT (auto-decided, never exposed):**

| ORT Option | Our Decision |
|------------|-------------|
| `graph_optimization_level` | Always optimize. Not configurable. |
| `inter_op_num_threads` | No inter-op parallelism. Deleted. |
| `enable_mem_pattern` | Always on. |
| `enable_cpu_mem_arena` | Always on. |
| `execution_mode` | Sequential only. |
| `optimized_model_filepath` | Don't export optimized models. |
| `log_severity_level` | Use `RUST_LOG` env var (tracing crate standard). |
| `providers` list | DevicePreference auto-selects. |
| `session_log_id` | Auto-generated. |
| `session_log_verbosity_level` | Use `RUST_LOG`. |

**Principle:** If the best value can be auto-determined, don't expose it.
If it must be exposed, namespace it clearly.

### 20.6 SessionBuilder

```rust
pub struct SessionBuilder {
    model_path: Option<PathBuf>,
    model_bytes: Option<Vec<u8>>,
    device: DevicePreference,
    memory_limit: Option<usize>,
    enable_profiling: bool,
    warmup_shapes: Vec<WarmupShape>,
    options: HashMap<String, String>,  // namespaced key-value
}

impl SessionBuilder {
    pub fn new() -> Self;
    pub fn model(self, path: impl AsRef<Path>) -> Self;
    pub fn model_bytes(self, bytes: &[u8]) -> Self;
    pub fn device(self, pref: DevicePreference) -> Self;
    pub fn memory_limit(self, bytes: usize) -> Self;
    pub fn profiling(self, enable: bool) -> Self;
    pub fn warmup(self, shapes: Vec<WarmupShape>) -> Self;

    /// Namespaced option. Unknown keys are rejected at build() time.
    pub fn option(self, key: &str, value: &str) -> Self;

    /// Build: load → detect device → optimize → compile → allocate.
    pub fn build(self) -> Result<InferenceSession>;
}
```

### 20.7 Auto-Detection Logic

```rust
fn auto_detect_device() -> Vec<Box<dyn ExecutionProvider>> {
    let mut eps: Vec<Box<dyn ExecutionProvider>> = vec![];
    if let Ok(cuda) = CudaEp::detect() { eps.push(Box::new(cuda)); }
    #[cfg(target_os = "macos")]
    if let Ok(mlx) = MlxEp::detect() { eps.push(Box::new(mlx)); }
    if let Ok(rocm) = detect_legacy_ep("libonnxruntime_rocm.so") { eps.push(rocm); }
    eps.push(Box::new(CpuEp::new()));  // always available
    eps
}
```

### 20.8 Python API

```python
import nxrt

# Zero-config:
session = nxrt.load("model.onnx")
output = session.run(input_ids=input_array)

# Device preference:
session = nxrt.load("model.onnx", device="gpu")

# Namespaced options:
session = nxrt.load("model.onnx",
    device="gpu:1",
    memory_limit=4 * 1024**3,
    options={
        "cuda.use_tf32": "true",
        "profiler.output": "/tmp/trace.json",
    },
)

# Zero-copy PyTorch:
import torch
tensor = torch.randn(1, 128, device="cuda")
output = session.run(input_ids=tensor)  # DLPack, no copy
torch_output = torch.from_dlpack(output["logits"])  # no copy
```

---

## 21. ORT C API Compatibility

### 21.1 Goal

Produce `libonnxruntime.so` that's a **drop-in replacement** for upstream ORT.
Python `onnxruntime`, Rust `ort` crate, OGA, C# — all work without code changes.

### 21.2 Implementation

```rust
/// Our OrtApi vtable — same layout as upstream ORT's.
#[repr(C)]
pub struct OrtApi {
    pub CreateEnv: unsafe extern "C" fn(OrtLoggingLevel, *const c_char, *mut *mut OrtEnv) -> *mut OrtStatus,
    pub CreateSession: unsafe extern "C" fn(*const OrtEnv, *const c_char, *const OrtSessionOptions, *mut *mut OrtSession) -> *mut OrtStatus,
    pub Run: unsafe extern "C" fn(*mut OrtSession, *const OrtRunOptions, *const *const c_char, *const *const OrtValue, usize, *const *const c_char, usize, *mut *mut OrtValue) -> *mut OrtStatus,
    pub CreateSessionOptions: unsafe extern "C" fn(*mut *mut OrtSessionOptions) -> *mut OrtStatus,
    pub AppendExecutionProvider: unsafe extern "C" fn(*mut OrtSessionOptions, *const c_char, *const *const c_char, *const *const c_char, usize) -> *mut OrtStatus,
    pub CreateTensorWithDataAsOrtValue: unsafe extern "C" fn(*const OrtMemoryInfo, *mut c_void, usize, *const i64, usize, ONNXTensorElementDataType, *mut *mut OrtValue) -> *mut OrtStatus,
    pub GetTensorData: unsafe extern "C" fn(*const OrtValue, *mut *const c_void) -> *mut OrtStatus,
    pub CreateIoBinding: unsafe extern "C" fn(*mut OrtSession, *mut *mut OrtIoBinding) -> *mut OrtStatus,
    pub RunWithBinding: unsafe extern "C" fn(*mut OrtSession, *const OrtRunOptions, *const OrtIoBinding) -> *mut OrtStatus,
    // ... ~150+ more functions (implemented incrementally)
}

/// Entry point — exported symbol.
#[no_mangle]
pub extern "C" fn OrtGetApiBase() -> *const OrtApiBase {
    &ORT_API_BASE
}

/// Shared library output:
/// libonnxruntime.so / libonnxruntime.dylib / onnxruntime.dll
```

```toml
# crates/onnx-runtime-capi/Cargo.toml
[lib]
name = "onnxruntime"
crate-type = ["cdylib"]
```

### 21.3 Incremental Implementation

- **Tier 1:** Session + Run + Tensor = basic Python onnxruntime inference
- **Tier 2:** IoBinding + SessionOptions + EP selection = OGA/advanced
- **Tier 3:** Custom ops, Allocator API, Training stubs

Unimplemented functions return `ORT_NOT_IMPLEMENTED` status.

### 21.4 EPContext Session Options (ORT-compatible)

To be a true drop-in, the C API must honor ORT's `EPContext`-generation session
options so ORT tooling (e.g. `onnxruntime.tools` context-cache dumping, QNN/OpenVINO
AOT scripts) works unchanged. These are set via `AddSessionConfigEntry` /
`AppendExecutionProvider` string key-values and are parsed into `SessionOptions`
(§20.5) fields consumed by the dump path (§57.4):

| Session-option key      | Type   | Default          | Meaning                                                              |
|-------------------------|--------|------------------|----------------------------------------------------------------------|
| `ep.context_enable`     | int    | `0`              | `1` = after compile, dump a context-cache `*_ctx.onnx` model.        |
| `ep.context_file_path`  | string | `<orig>_ctx.onnx`| Output path for the generated context model.                         |
| `ep.context_embed_mode` | int    | `0`              | `0` = write external file, store its path; `1` = embed blob in node. |

> **Two distinct `embed_mode` defaults — do not conflate them.** The *session
> option* `ep.context_embed_mode` that drives **generation** defaults to **`0`**
> (external file) in ORT (`ep_context_options.cc`). The *op attribute*
> `embed_mode` baked into an on-disk `EPContext` node defaults to **`1`**
> (payload inline) when the attribute is absent (§57.2). In other words: when you
> ask ORT to *generate* a context model without specifying a mode, it writes an
> **external** blob; but when you *read* an EPContext node whose `embed_mode`
> attribute is missing, you assume the payload is **inline**. The two `1`↔`0`
> defaults are intentional and independent.

```rust
// Parsed in the capi layer, stored on SessionOptions, read by the writer (§57.4).
pub struct EpContextGenOptions {
    pub enable: bool,           // ep.context_enable
    pub file_path: Option<PathBuf>, // ep.context_file_path (default: <orig>_ctx.onnx)
    pub embed_mode: EmbedMode,  // ep.context_embed_mode → Embedded | ExternalFile
                                //   (session-option default = ExternalFile, i.e. 0)
}
```

Unknown/unsupported EP keys are ignored (never error), matching ORT semantics.

---

## 22. Error Types

```rust
/// Top-level error type for the runtime.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    // === Model loading ===
    #[error("Failed to parse ONNX protobuf: {0}")]
    ProtobufParse(#[from] prost::DecodeError),
    #[error("Unsupported opset: domain={domain}, version={version}")]
    UnsupportedOpset { domain: String, version: u64 },
    #[error("External data file not found: {path}")]
    ExternalDataNotFound { path: PathBuf },
    #[error("Weight mmap failed: {0}")]
    MmapError(#[from] std::io::Error),

    // === Graph ===
    #[error("Graph validation failed: {0:?}")]
    GraphInvalid(Vec<GraphError>),
    #[error("Cycle detected in graph")]
    CycleDetected,

    // === EP / Kernel ===
    #[error("No EP supports op {op_type} on any available device")]
    NoEpForOp { op_type: String },
    #[error("Kernel execution failed: {0}")]
    KernelFailed(String),
    #[error("EP panicked during execution")]
    EpPanicked,
    #[error("EP plugin load failed: {path}: {reason}")]
    EpLoadFailed { path: PathBuf, reason: String },

    // === Memory ===
    #[error("Device OOM: requested {requested} bytes, available {available}")]
    OutOfMemory { requested: usize, available: usize },
    #[error("Allocation alignment mismatch")]
    AlignmentError,

    // === Placement ===
    #[error("ILP solver failed: {0}")]
    PlacementFailed(String),

    // === Shape ===
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch { expected: Vec<usize>, actual: Vec<usize> },
    #[error("Broadcast incompatible: {a:?} vs {b:?}")]
    BroadcastIncompatible { a: Vec<usize>, b: Vec<usize> },

    // === Runtime ===
    #[error("Session not initialized")]
    SessionNotInitialized,
    #[error("Input not found: {name}")]
    InputNotFound { name: String },
    #[error("CUDA error: {0}")]
    CudaError(String),
}

/// ORT-compatible status code for C API layer.
#[repr(C)]
pub enum OrtErrorCode {
    Ok = 0,
    Fail = 1,
    InvalidArgument = 2,
    NoSuchFile = 3,
    NoModel = 4,
    EngineMismatch = 5,
    InvalidProtobuf = 6,
    ModelLoaded = 7,
    NotImplemented = 8,
    InvalidGraph = 10,
    EpFail = 11,
}

impl From<&Error> for OrtErrorCode {
    fn from(e: &Error) -> Self {
        match e {
            Error::ProtobufParse(_) => Self::InvalidProtobuf,
            Error::NoEpForOp { .. } => Self::EpFail,
            Error::OutOfMemory { .. } => Self::Fail,
            _ => Self::Fail,
        }
    }
}
```

---

## 23. Crate Structure

```
onnx-genai/                               (monorepo)
├── crates/
│   │
│   │  ── Runtime layer (new — ORT 2.0) ──
│   ├── onnx-runtime-ir/                         # Graph IR, types, shapes, strides, layout
│   │                                     # (named onnx-runtime-ir because onnx-ir is taken)
│   ├── onnx-runtime-loader/                       # ONNX protobuf → IR, weight mmap
│   ├── onnx-runtime-optimizer/                    # Optimization passes pipeline
│   ├── onnx-runtime-cost-model/                   # Cost estimation, calibration
│   ├── onnx-runtime-memory/                       # Arena allocator, memory planner
│   ├── onnx-runtime-scheduler/                    # Async DAG executor, streams, fences
│   ├── onnx-runtime-ep-api/                       # ExecutionProvider trait + ORT ABI bridge
│   ├── onnx-runtime-ep-cpu/                       # CPU EP (oneDNN, C++ FFI) — we maintain
│   ├── onnx-runtime-ep-cuda/                      # CUDA EP (cudarc + cuBLASLt + CuTe; §15) — we maintain
│   ├── onnx-runtime-session/                      # Session builder, inference API
│   ├── onnx-runtime-tracer/                       # Unified tracing (shared by runtime + genai)
│   ├── onnx-runtime-capi/                         # C ABI: libonnxruntime.so drop-in
│   ├── onnx-runtime-autotuner/                    # Agent-driven optimization loop
│   │
│   │  ── GenAI layer (existing) ──
│   ├── onnx-genai/                      # Main crate / facade
│   ├── onnx-genai-engine/               # Batching, speculative, pipeline
│   ├── onnx-genai-kv/                   # KV cache (paged, tiered, heterogeneous)
│   ├── onnx-genai-server/               # OpenAI HTTP API
│   ├── onnx-genai-router/               # Multi-node routing
│   ├── onnx-genai-scheduler/            # GenAI scheduling
│   ├── onnx-genai-ort/                  # ORT C API bindings (current backend)
│   ├── onnx-genai-metadata/             # inference_metadata.yaml schema
│   ├── onnx-genai-preprocess/           # Image/tokenizer preprocessing
│   ├── onnx-genai-genai-config/         # GenAI config
│   └── onnx-genai-bench/               # Benchmark tools
│
├── native-eps/
│   ├── cpu/                              # C++ CPU kernels (oneDNN FFI)
│   └── cuda/                             # C++ CUDA kernels (CuTe + cuBLAS FFI)
│
├── bindings/
│   └── python/                           # PyO3: nxrt + per-EP packages
│
├── docs/
│   ├── DESIGN.md                         # GenAI layer design
│   ├── ORT2.md                           # This document
│   └── PROGRESS.md
│
└── Cargo.toml                            # Workspace

EP compatibility:
  - onnx-runtime-ep-cpu, onnx-runtime-ep-cuda: we write and maintain (ported from ORT C++)
  - MLX EP: Justin's existing implementation (separate or merged later)
  - QNN, OpenVINO, WebGPU, CoreML, ROCm, etc.: loaded as legacy ORT
    plugin EPs via dlopen + C ABI bridge. We don't write these —
    we just implement the host-side ABI so they load and run.
```

### 23.1 Backend Feature Flag

GenAI crates are backend-agnostic:

```toml
# In onnx-genai-engine/Cargo.toml
[features]
default = ["backend-ort"]
backend-ort = ["dep:onnx-genai-ort"]      # upstream ORT via C API
backend-nxrt = ["dep:onnx-runtime-session"]        # our runtime
```

```rust
/// Backend trait. Both ORT wrapper and our runtime implement this.
pub trait InferenceBackend: Send + Sync {
    type Session: BackendSession;
    fn load_model(&self, path: &Path, options: &SessionOptions) -> Result<Self::Session>;
}

pub trait BackendSession: Send {
    fn run(&mut self, inputs: &[Tensor], outputs: &mut [Tensor]) -> Result<()>;
    fn io_binding(&mut self) -> Result<IoBinding<'_>>;
}
```

---

## 24. Python Bindings

### 24.1 Main Package: `nxrt`

```python
import nxrt
from ort_ep_cuda import CudaEp

# Create session with our runtime
session = nxrt.InferenceSession(
    "model.onnx",
    providers=[CudaEp(device_id=0), nxrt.CpuEp()],
    optimization_level="aggressive",
)

# Run inference
outputs = session.run({"input_ids": input_array})

# Profile
report = session.profile({"input_ids": input_array}, num_runs=10)
print(report.bottlenecks)

# Auto-tune
tuner = nxrt.AutoTuner(session)
result = tuner.auto_tune({"input_ids": input_array}, max_iterations=20)
```

### 24.2 Per-EP Packages

Each EP is a separate pip package:
- `pip install onnx-runtime-ep-cuda`
- `pip install onnx-runtime-ep-mlx`
- `pip install onnx-runtime-ep-webgpu`

```python
# Each EP package exports:
# - provider() → can be passed to InferenceSession
# - library_path() → .so path for ORT compatibility
import ort_ep_cuda
print(ort_ep_cuda.library_path())  # /path/to/ort_ep_cuda.so
```

### 24.3 ORT Compatibility Mode

```python
# Drop-in replacement for onnxruntime Python package:
# Set LD_PRELOAD or replace libonnxruntime.so in site-packages
import onnxruntime as ort  # unchanged user code
session = ort.InferenceSession("model.onnx")  # uses our runtime transparently
```

---

## 25. Platform Support

### 25.1 Platform × EP Matrix

| Platform | EP crates available | Weight mmap | CUDA Graph | Notes |
|----------|-------------------|-------------|------------|-------|
| Linux x64 | cpu, cuda, rocm, openvino, qnn, webgpu | ✅ | ✅ | Primary |
| macOS arm64 | cpu, mlx, coreml, webgpu | ✅ | ❌ | MLX for GPU |
| Windows x64 | cpu, cuda, openvino, webgpu | ✅ | ✅ | |
| Linux arm64 | cpu, qnn, webgpu | ✅ | ❌ | Edge / server |
| Android (ARM) | cpu-mobile, qnn, webgpu | ✅ | ❌ | XNNPACK backend |
| iOS (ARM) | cpu-apple, coreml, metal | ✅ | ❌ | Accelerate + Metal |
| Web (WASM) | webgpu | ❌ (fetch) | ❌ | wasm-bindgen |

### 25.2 CPU Backend Strategy

| Target | Backend | Rationale |
|--------|---------|----------|
| x86 (Intel/AMD) | **oneDNN** | Best-in-class. AMX/AVX-512/VNNI. Battle-tested. |
| ARM server (Neoverse) | **oneDNN** | Has NEON/SVE paths. Single dep covers x86+ARM server. |
| ARM mobile (Android) | **XNNPACK** | Lightweight (~1MB), int8 optimized, low startup. oneDNN too heavy for mobile. |
| Apple Silicon | **Accelerate** (vDSP/BNNS) | System framework, zero-dep, Apple-tuned. |
| Fallback | **Generic Rust** | Pure Rust kernels. Slow but compiles anywhere. Correctness baseline. |

**oneDNN scope:** Covers x86 + ARM server (Linux/Windows). We link it statically in the
cpu EP crate. Mobile and Apple go through separate lighter backends.

**XNNPACK scope:** Android only. Google-maintained, TFLite's CPU backend.
Small binary, fast startup, excellent int8/fp16 on Cortex-A cores.

```rust
// crates/onnx-runtime-ep-cpu/src/backend.rs
pub enum CpuBackend {
    OneDnn,       // x86 + ARM server
    Xnnpack,      // Android mobile
    Accelerate,   // macOS / iOS
    Generic,      // fallback
}

impl CpuBackend {
    pub fn auto_detect() -> Self {
        #[cfg(target_os = "android")]
        { Self::Xnnpack }
        #[cfg(target_os = "macos")]
        { Self::Accelerate }
        #[cfg(target_os = "ios")]
        { Self::Accelerate }
        #[cfg(all(not(target_os = "android"), not(target_os = "macos"), not(target_os = "ios")))]
        {
            if has_onednn() { Self::OneDnn } else { Self::Generic }
        }
    }
}
```

### 25.3 GPU Backend Strategy

| Target | EP | Approach | Status |
|--------|-----|---------|--------|
| NVIDIA | `onnx-runtime-ep-cuda` | cuBLAS + CuTe + FA3 (§15) | Designed |
| Apple GPU | `onnx-runtime-ep-mlx` | MLX kernels, Metal shaders | **183/202 ops** ✅ |
| AMD GPU (ROCm) | `onnx-runtime-ep-rocm` | **hipify from CUDA EP** | Planned |
| Intel GPU (Arc/Xe) | **OpenVINO plugin EP** | Don't write ourselves | Plugin |
| Mobile GPU (Mali/Adreno) | **QNN plugin EP** or WebGPU | Don't write ourselves | Plugin |

### 25.4 ROCm Strategy: Hipify, Don't Rewrite

ROCm EP = CUDA EP mechanically translated. Minimize manual work:

```bash
# hipify-perl converts CUDA API calls → HIP API calls
hipify-perl ep-cuda/src/kernels/*.cu → ep-rocm/src/kernels/*.cu

# API mapping (1:1 in most cases):
# cuBLAS      → hipBLAS (same API shape)
# cudaMalloc  → hipMalloc
# cudaStream  → hipStream
# CUPTI       → rocTracer (profiling)
# cuDNN       → MIOpen (not 1:1, needs adaptation)
# CUTLASS     → composable_kernel (AMD's equivalent, needs manual port)
```

**What hipifies cleanly (auto-convert):**
- Memory management (malloc/free/memcpy)
- Stream management
- cuBLAS GEMM calls → hipBLAS
- Simple CUDA C kernels (elementwise, reduction)
- Launch configuration (grid, block)

**What needs manual work:**
- cuDNN SDPA → MIOpen attention (different API surface)
- CuTe/CUTLASS kernels → composable_kernel or hand-port
- FlashAttention → use AMD's [flash-attention fork](https://github.com/ROCm/flash-attention)
- CUPTI → rocTracer (different profiling model)

**Implementation plan:**
```
Phase 1: hipify basic kernels + hipBLAS GEMM → run BERT on ROCm
Phase 2: MIOpen attention → run LLM decode
Phase 3: AMD flash-attention fork → full LLM serving
```

**Build system:**
```rust
// crates/onnx-runtime-ep-rocm/build.rs
fn main() {
    // hipcc compiles .cu files (HIP understands CUDA syntax)
    let hipcc = env::var("HIPCC").unwrap_or_else(|_| "hipcc".into());
    for src in glob("src/kernels/*.cu") {
        Command::new(&hipcc).args([
            "-O3", "--std=c++17",
            "--offload-arch=gfx942",  // MI300X
            "--offload-arch=gfx90a",  // MI250X
            "-c", &src, "-o", &obj,
        ]).status().unwrap();
    }
    println!("cargo:rustc-link-lib=static=hip_kernels");
    println!("cargo:rustc-link-lib=dylib=hipblas");
    println!("cargo:rustc-link-lib=dylib=amdhip64");
}
```

**Goal:** Keep ep-rocm as a thin translation layer over ep-cuda. When CUDA EP gets a
new kernel, running hipify + minimal fixup produces the ROCm version. No parallel
development — **CUDA leads, ROCm follows mechanically.**

### 25.5 Apple Strategy: MLX EP (Already In Progress)

MLX EP is authored separately (183/202 ai.onnx ops complete). It uses:
- Metal compute shaders for GPU kernels
- Metal Performance Shaders (MPS) for GEMM/Conv
- Unified memory (no explicit H2D/D2H — Apple's advantage)

CPU path on Apple: Accelerate framework (BNNS for NN ops, vDSP for signal).

### 25.6 Plugin EPs (Don't Write, Just Load)

These EPs are maintained by hardware vendors. We preserve ORT's plugin ABI (§21)
so they load without modification:

| Plugin EP | Vendor | Targets | Notes |
|-----------|--------|---------|-------|
| OpenVINO | Intel | Intel CPU/GPU/NPU | Best path for Intel discrete GPU (Arc) |
| QNN | Qualcomm | Snapdragon NPU/GPU | Android mobile AI |
| CoreML | Apple | Apple NPU (ANE) | iOS/macOS neural engine |
| TensorRT | NVIDIA | NVIDIA GPU | Alternative to our CUDA EP (user choice) |
| WebGPU | W3C | Browser + cross-platform | Portable but slower |

We don't write these — we ensure our EP plugin interface (§4, §21) is compatible
so vendor-maintained shared libraries work unchanged.

### 25.7 Design Decisions

| Decision | Choice | Rationale |
|----------|--------|----------|
| CPU: single backend vs multi | **Multi** (oneDNN/XNNPACK/Accelerate) | No single lib covers all targets well |
| Mobile CPU | **XNNPACK** not oneDNN | oneDNN binary too large, startup too slow for mobile |
| Apple CPU | **Accelerate** not oneDNN | System framework, zero dep, Apple-tuned |
| ROCm approach | **Hipify from CUDA** | Minimize parallel development. CUDA leads. |
| ROCm attention | **AMD flash-attention fork** | Don't hand-write; AMD maintains their own |
| Intel GPU | **OpenVINO plugin** | Intel maintains it; we just load it |
| Mobile GPU | **QNN plugin / WebGPU** | Fragmented landscape; vendor plugins win |
| Plugin EP compat | **ORT ABI compatible** | Reuse existing vendor effort |

---

## 26. Safety and Failure Handling

### 26.1 FFI Boundary

```rust
pub fn safe_ffi_call<F, T>(f: F) -> Result<T>
where F: FnOnce() -> T + std::panic::UnwindSafe {
    std::panic::catch_unwind(f).map_err(|_| Error::EpPanicked)
}
```

### 26.2 EP Fallback

If an EP fails at runtime:
1. Log failure with full context
2. Mark failing kernel as "poisoned"
3. Re-plan without failed EP
4. Continue on fallback device

### 26.3 Thread Safety

- Graph IR: immutable after optimization → `Arc<Graph>`
- Weights: read-only mmap → `Arc<WeightStore>`
- Execution state: single-writer per session (no locks)
- EP calls: serialized per-EP instance
- Transfer scheduler: owns streams, channel-based communication

---

## 27. Testing Strategy

### 27.1 Unit Tests (per crate)

```rust
// onnx-runtime-ir: graph construction, topological sort, validation
// onnx-runtime-optimizer: each pass in isolation with small test graphs
// onnx-runtime-cost-model: cost formula correctness
// onnx-runtime-memory: arena allocation, aliasing correctness
// onnx-runtime-scheduler: DAG execution ordering, fence semantics
```

### 27.2 Integration Tests

```rust
// Load real models from ONNX model zoo, run inference, check output shape/dtype.
// Models: BERT, ResNet50, GPT-2, Llama-7B (quantized), Whisper
#[test]
fn test_bert_inference() {
    let session = SessionBuilder::new()
        .with_model_path("tests/models/bert-base.onnx")
        .with_ep(Box::new(CpuEp::new()))
        .build().unwrap();
    let output = session.run(&[("input_ids", &input)]).unwrap();
    assert_eq!(output[0].shape(), &[1, 128, 768]);
}
```

### 27.3 Conformance Testing

```rust
/// Compare our output against upstream ORT's output for the same model + inputs.
/// Tolerance: fp32 atol=1e-5, fp16 atol=1e-3.
pub fn conformance_test(model_path: &Path, inputs: &[Tensor], tolerance: f64) -> Result<()> {
    let ort_output = run_with_upstream_ort(model_path, inputs)?;
    let our_output = run_with_nxrt(model_path, inputs)?;
    for (ort_t, our_t) in ort_output.iter().zip(our_output.iter()) {
        assert_tensors_close(ort_t, our_t, tolerance)?;
    }
    Ok(())
}
```

### 27.4 Fuzzing

```rust
// Fuzz the ONNX loader with arbitrary protobuf bytes
// Fuzz the shape inference with random shapes
// Fuzz the memory planner with random lifetime intervals
// Target: no panics, no UB, graceful errors
```

---

## 28. Resolved Decisions

| Decision | Resolution | Rationale |
|----------|-----------|----------|
| Training support | **No** | Out of scope. Inference-only runtime. |
| Graph optimization levels | **No** | Always optimize. Single default pass pipeline. No user-facing knob. |
| Parallel execution (inter-op) | **No** | ORT proved it doesn't help in practice; adds complexity for no gain. |
| Shape inference | **Port from Python** | Justin has a working Python impl; port to Rust when ready. |
| ONNX external data | **Yes, required** | All large models use it. Mandatory from Phase 1. |
| Custom Ops | **Yes, via C ABI** | Support ORT Extensions registration mechanism through the same C ABI bridge. |

| Execution hints | **Yes** | Users can provide placement/memory hints via builder, options, or model metadata. Hints bias the optimizer (soft preference) unless marked as Force (hard constraint). |
| IoBinding as user API | **No** | Buffer reuse is internal. Session auto-reuses output buffers on stable shapes, auto-captures CUDA graph after 3 stable runs. Users use DLPack for explicit buffer control. |
| Precompiled plans | **Yes** | AOT compilation à la ExecuTorch: partition + optimize + serialize plan. Instant reload without re-optimization. |
| AOT memory plan | **Yes** | Pre-compute tensor offsets at compile time. Runtime = one malloc + offset table. Zero allocation overhead. |
| Quantization as EP concern | **Yes** | EPs handle quantized tensors natively (fused dequant+compute). No separate quantization pass. |
| TensorRT-RTX EP as test target | **Yes** | NVIDIA's [TensorRT-RTX EP](https://github.com/NVIDIA/TensorRT-RTX-EP-ABI) — ABI-stable ORT plugin EP for RTX GPUs. Use as compatibility test target for our legacy EP loading (dlopen + C ABI bridge). If we can load and run TRT-RTX EP unchanged, our ABI bridge is correct. |

## 29. Open Questions

1. **JIT compilation** — Cranelift/LLVM for custom fused kernels? Or leave to EPs?
2. **Model format** — Own optimized format for faster loading? Or always from ONNX?
3. ~~**Minimum opset** — Opset 17+ (modern LLMs) vs opset 7+ (full ORT compat)?~~ **Resolved: opset 17 minimum.**
4. **Tensor parallelism** — Built into runtime or left to GenAI layer?
5. **Disaggregated prefill/decode** — Runtime-level support or application-level?

---

## 30. Memory Tiering & Offloading

**Design principle: VRAM is a cache, not a hard requirement. Any model runs on any hardware — only speed differs.**

### 30.1 Problem

ORT today: model doesn't fit in GPU → OOM crash. Unacceptable.

Our guarantee: **never OOM.** Auto-degrade to slower tiers when VRAM is insufficient.

### 30.2 Memory Hierarchy

```
Tier 0: GPU HBM     — ~2 TB/s bandwidth, 16-80 GB capacity
Tier 1: CPU DRAM    — ~50 GB/s bandwidth, 64-512 GB capacity
Tier 2: NVMe/Disk   — ~7 GB/s bandwidth, TB+ capacity (mmap)
```

The runtime treats these as a unified address space with different costs.

### 30.3 Architecture

```rust
pub struct TieredMemoryManager {
    /// Tiers ordered by speed (fastest first).
    tiers: Vec<MemoryTier>,
    /// Weight placement: which tier each tensor lives on.
    placement: HashMap<TensorId, TierPlacement>,
    /// Prefetch queue: async bring-to-GPU requests.
    prefetch_queue: VecDeque<PrefetchRequest>,
    /// Memory pressure monitor.
    pressure: MemoryPressureMonitor,
}

pub struct MemoryTier {
    kind: TierKind,             // GpuHbm, CpuDram, Disk
    capacity_bytes: usize,
    used: AtomicUsize,
    bandwidth_gbps: f64,        // for cost model
    allocator: Box<dyn Allocator>,
}

pub enum TierPlacement {
    /// Fits in GPU. Happy path, zero overhead.
    Resident { device: DeviceId },
    /// In CPU RAM, streamed to GPU on demand.
    Offloaded { host_ptr: *mut u8, size: usize },
    /// On disk, mmap'd on demand (extreme case: 405B on laptop).
    DiskBacked { path: PathBuf, offset: u64, size: usize },
}
```

### 30.4 Three Offloading Mechanisms

#### A. Weight Offloading (static, decided at load time)

On model load, place weights by priority until GPU budget is filled:

```rust
fn plan_weight_placement(model: &Model, gpu_budget: usize) -> PlacementPlan {
    let mut weights: Vec<_> = model.weights()
        .map(|w| (w.id, w.size, compute_priority(w)))
        .collect();
    // Priority: attention weights > MLP weights > embeddings > lm_head
    weights.sort_by(|a, b| b.2.cmp(&a.2));

    let mut gpu_used = 0;
    let mut plan = PlacementPlan::new();

    for (id, size, _priority) in &weights {
        if gpu_used + size <= gpu_budget {
            plan.place(*id, Tier::Gpu);
            gpu_used += size;
        } else {
            plan.place(*id, Tier::Cpu);  // spill to CPU
        }
    }
    plan
}
```

Priority heuristic (customizable via `execution_hints.json` or `onnx_runtime.memory.priority`):
1. Attention Q/K/V projections (hot in decode loop)
2. MLP gate/up/down projections
3. Embedding table (one lookup per token, memory-bound anyway)
4. LM head (only used once at end)

#### B. Activation Offloading (dynamic, during execution)

For long-sequence prefill where activations exceed GPU memory:

```rust
fn execute_with_activation_spill(&mut self, layers: &[Layer]) -> Result<()> {
    for (i, layer) in layers.iter().enumerate() {
        // Async prefetch next layer's weights (overlaps with current compute)
        if i + 1 < layers.len() {
            self.prefetch_weights_async(i + 1);
        }

        // Execute current layer
        let activation = layer.execute(&self.current_activation)?;

        // If memory pressure high, spill activation to CPU
        if self.pressure.gpu_utilization() > 0.90 {
            self.spill_activation_to_cpu(&activation);
        }

        self.current_activation = activation;
    }
    Ok(())
}
```

#### C. KV Cache Offloading (incremental, per-page eviction)

Long context: KV cache is the biggest VRAM consumer. Page-granularity eviction:

```rust
impl PagedKvCache {
    /// Evict least-recently-used pages to CPU when GPU pages exhausted.
    fn ensure_capacity(&mut self, pages_needed: usize) {
        while self.gpu_pages_free() < pages_needed {
            let victim = self.lru_page();
            self.async_move_page_to_cpu(victim);
        }
    }

    /// Before attention: prefetch needed KV pages back to GPU.
    fn prefetch_for_attention(&mut self, page_indices: &[PageId]) {
        for &page_id in page_indices {
            if self.page_location(page_id) == Tier::Cpu {
                self.async_move_page_to_gpu(page_id);
            }
        }
    }
}
```

### 30.5 Overlap: Hiding Transfer Latency

Offloading doesn't mean stalling — **prefetch overlaps with compute:**

```
Timeline (layer-pipeline offloading):

GPU compute:  [== Layer N ==]  [== Layer N+1 ==]  [== Layer N+2 ==]
H2D stream:        [-- prefetch N+1 weights --]  [-- prefetch N+2 --]
D2H stream:   [-- spill N-1 activation --]

If compute time > transfer time: zero visible overhead.
If not: partially hidden, still better than blocking.
```

Implementation: separate CUDA streams for compute vs transfer, fence synchronization:

```rust
pub struct OverlappedExecutor {
    compute_stream: CudaStream,
    h2d_stream: CudaStream,      // host-to-device transfers
    d2h_stream: CudaStream,      // device-to-host spills
}

impl OverlappedExecutor {
    fn execute_layer(&mut self, layer: usize) {
        // Fence: wait for prefetch of this layer's weights
        self.h2d_stream.record_event(&self.prefetch_done[layer]);
        self.compute_stream.wait_event(&self.prefetch_done[layer]);

        // Compute on compute stream
        self.compute_stream.launch_kernels(&self.layers[layer]);

        // Async prefetch next layer (doesn't block compute)
        self.h2d_stream.copy_h2d(&self.weights[layer + 1]);

        // Async spill current activation if needed
        if self.should_spill() {
            self.d2h_stream.copy_d2h(&self.activations[layer]);
        }
    }
}
```

### 30.6 Performance Expectation

| Scenario | Relative Speed | What's happening |
|----------|---------------|------------------|
| 100% GPU-resident | 1.0× | Ideal |
| 30% weights on CPU | ~0.7× | Prefetch mostly hidden by compute |
| 70% weights on CPU | ~0.3-0.4× | Transfer becomes bottleneck |
| Activation spill | ~0.5× | Extra D2H + H2D per layer |
| KV pages on CPU | ~0.7-0.9× | Only evicted pages need fetch |
| 100% CPU (no GPU) | ~0.05-0.1× | Fallback, but it runs |

### 30.7 User API

```python
# Zero-config: auto-detect GPU capacity, auto-offload if needed
session = nxrt.load("llama-70b.onnx")  # 16GB GPU? Fine. Auto-offloads.

# Explicit memory limit (leave room for other processes)
session = nxrt.load("model.onnx", memory_limit=12 * GB)

# Force full offload (CPU-only mode)
session = nxrt.load("model.onnx", device="cpu")
```

Namespaced options for fine control:
```python
session = nxrt.load("model.onnx", options={
    "memory.gpu_budget_mb": "12000",       # 12GB GPU budget
    "memory.offload_strategy": "layerwise", # vs "tensorwise"
    "memory.prefetch_layers": "2",          # prefetch 2 layers ahead
    "memory.kv_gpu_pages": "1024",          # max KV pages on GPU
    "memory.activation_spill": "auto",      # auto/always/never
})
```

### 30.8 Design Choices

| Choice | Decision | Rationale |
|--------|----------|----------|
| Never OOM | **Yes** | Core guarantee. Runtime degrades gracefully. |
| Weight placement at load time | **Yes (static)** | Avoids runtime jitter. Re-plan only on explicit resize. |
| Activation spill | **Dynamic** | Can't predict at load time (depends on input shapes). |
| KV cache eviction | **LRU per-page** | Matches paged attention. Old context evicted first. |
| Overlap compute + transfer | **Always** | Separate streams, fence sync. Zero overhead when compute-bound. |
| Disk tier (mmap) | **Tier 2 fallback** | For 405B on 32GB RAM laptops. Not primary path. |
| Offload granularity | **Per-layer (weights), per-page (KV), per-tensor (activation)** | Each has different lifecycle. |

---

## 31. Multi-Session Resource Coordination & Multi-GPU

### 31.1 Problem

Multiple sessions compete for GPU resources:
- User loads Llama-70B (chat) + Whisper (transcription) + SDXL (image gen)
- Multi-GPU systems need intelligent placement
- Single large models need sharding across devices

ORT today: each session is isolated, no coordination. OOM if combined usage exceeds device.

### 31.2 Resource Broker (Global Coordinator)

```rust
/// Global singleton — coordinates all sessions on this host.
pub struct ResourceBroker {
    /// All active sessions and their resource claims.
    sessions: RwLock<HashMap<SessionId, ResourceClaim>>,
    /// Per-device state (capacity, usage, temperature).
    devices: Vec<DeviceState>,
    /// Scheduling policy.
    policy: AllocationPolicy,
    /// Communication backends (NCCL, Gloo).
    comm: Box<dyn CommBackend>,
}

pub struct ResourceClaim {
    session_id: SessionId,
    priority: SessionPriority,
    gpu_bytes_used: usize,
    placement: PlacementPlan,
    active: bool,
    last_active: Instant,
}

pub enum SessionPriority {
    /// Real-time interactive (chat decode). Never preempt.
    Realtime,
    /// Foreground batch (user waiting).
    Foreground,
    /// Background (async generation).
    Background,
    /// Idle (loaded, not running). First to evict.
    Idle,
}

pub enum AllocationPolicy {
    /// First come first served.
    Fcfs,
    /// Priority-based preemption. High steals from low.
    Priority,
    /// Fair share. Proportional budget per session.
    FairShare,
    /// Weighted allocation.
    Weighted { weights: HashMap<SessionId, f32> },
}
```

**Preemption logic (priority-based):**
```rust
impl ResourceBroker {
    fn reclaim_memory(&mut self, needed: usize, requestor: &ResourceClaim) -> Result<()> {
        // 1. Evict idle sessions (offload to CPU)
        // 2. Shrink background sessions (reduce GPU budget)
        // 3. If requestor is Realtime, preempt Foreground
        // 4. Requestor itself offloads (never OOM)
        Ok(())
    }
}
```

### 31.3 Multi-GPU: Parallel Strategies

Three orthogonal strategies, composable:

```rust
pub enum ParallelStrategy {
    /// Tensor Parallelism: split weight matrices across GPUs.
    /// Each GPU holds a slice of every layer. AllReduce after parallel regions.
    Tensor { degree: usize },

    /// Pipeline Parallelism: different layers on different GPUs.
    /// Micro-batching hides pipeline bubbles.
    Pipeline { stages: Vec<LayerRange> },

    /// Data Parallelism: model replicated, different inputs per replica.
    /// For throughput (batch serving), not single-request latency.
    Data { replicas: usize },

    /// Hybrid: TP within node + PP across nodes (Megatron-style).
    Hybrid { tp_degree: usize, pp_stages: Vec<LayerRange> },
}
```

**Tensor Parallelism (most common for LLM inference):**
```
Original:  Q = X @ W_q         (X: [B,S,H], W_q: [H,H])
TP=2:      Q_0 = X @ W_q[:,:H/2]   → GPU 0
           Q_1 = X @ W_q[:,H/2:]   → GPU 1
           ... attention independently per shard ...
           O = AllReduce(O_0, O_1)  → synchronization point
```

### 31.4 IR Sharding Annotations

Our IR supports per-tensor sharding specs:

```rust
/// How a tensor is distributed across devices.
#[derive(Clone, Debug)]
pub enum ShardingSpec {
    /// Not sharded. Full tensor on one device.
    Replicated,
    /// Split along one axis across devices.
    Split {
        axis: usize,
        num_shards: usize,
        device_map: Vec<DeviceId>,
    },
    /// Partial result needing collective to materialize (after TP matmul).
    Partial {
        reduce_op: ReduceOp,
        devices: Vec<DeviceId>,
    },
}

/// Sharding annotation on an IR node.
pub struct NodeSharding {
    input_specs: Vec<ShardingSpec>,
    output_specs: Vec<ShardingSpec>,
}

/// Communication ops inserted by the sharding pass.
pub enum CommOp {
    AllReduce { group: Vec<DeviceId>, op: ReduceOp },
    AllGather { axis: usize, group: Vec<DeviceId> },
    ReduceScatter { axis: usize, group: Vec<DeviceId>, op: ReduceOp },
    PipelineSend { from: DeviceId, to: DeviceId, tensor: TensorId },
    PipelineRecv { from: DeviceId, to: DeviceId, shape: Shape },
}
```

### 31.5 Sharding Pass (Optimizer)

Automatic sharding: given a strategy, annotate IR and insert communication ops:

```rust
pub struct ShardingPass {
    strategy: ParallelStrategy,
    devices: Vec<DeviceId>,
}

impl OptimizerPass for ShardingPass {
    fn run(&self, graph: &mut Graph) -> Result<()> {
        match &self.strategy {
            ParallelStrategy::Tensor { degree } => {
                // Split MatMul weights along columns (or rows)
                // Insert AllReduce after parallel matmul
                for node in graph.nodes_by_type("MatMul") {
                    self.shard_matmul(graph, node, *degree)?;
                }
            }
            ParallelStrategy::Pipeline { stages } => {
                // Partition graph by layer range
                // Insert Send/Recv at stage boundaries
                for (i, range) in stages.iter().enumerate() {
                    self.assign_stage(graph, range, self.devices[i])?;
                }
            }
            ParallelStrategy::Data { .. } => {
                // No IR change — scheduler handles replication
            }
            _ => {}
        }
        Ok(())
    }
}
```

### 31.6 Communication Backend

```rust
pub trait CommBackend: Send + Sync {
    fn all_reduce(&self, tensor: &mut Tensor, op: ReduceOp, group: &CommGroup) -> Result<()>;
    fn all_gather(&self, input: &Tensor, output: &mut Tensor, group: &CommGroup) -> Result<()>;
    fn reduce_scatter(&self, tensor: &mut Tensor, op: ReduceOp, group: &CommGroup) -> Result<()>;
    fn send(&self, tensor: &Tensor, dest: DeviceId) -> Result<()>;
    fn recv(&self, tensor: &mut Tensor, src: DeviceId) -> Result<()>;
}

/// NCCL for multi-GPU on same node (NVLink/PCIe).
pub struct NcclBackend { /* ncclComm_t per device pair */ }

/// Gloo for CPU fallback / multi-node.
pub struct GlooBackend { /* ... */ }
```

### 31.7 Multi-Model on Multi-GPU (Placement)

ResourceBroker assigns sessions to devices:

```rust
impl ResourceBroker {
    fn assign_devices(&self, session: &SessionBuilder) -> Vec<DeviceId> {
        let model_size = session.estimated_memory();
        // Strategy: bin-pack models onto fewest GPUs
        // or spread for thermal/bandwidth balance
        self.bin_pack_or_spread(model_size)
    }
}
```

Example: 4x RTX 4090 (24GB each)
```
Llama-70B (TP=4):  GPU 0-3 each hold 1/4 of weights
Llama-8B:          GPU 0 (fits entirely, shares with Llama-70B shard)
SDXL:              GPU 1 (background, preemptable)
```

### 31.8 User API

```python
import nxrt

# Auto: single model, best available device(s)
session = nxrt.load("model.onnx")

# Tensor parallelism (split across 4 GPUs)
session = nxrt.load("llama-70b.onnx", options={
    "parallel.strategy": "tensor",
    "parallel.degree": "4",
})

# Pipeline parallelism (explicit layer→GPU mapping)
session = nxrt.load("llama-70b.onnx", options={
    "parallel.strategy": "pipeline",
    "parallel.stages": "0-15:gpu:0,16-31:gpu:1",
})

# Data parallelism (throughput mode for serving)
session = nxrt.load("model.onnx", options={
    "parallel.strategy": "data",
    "parallel.replicas": "4",
})

# Multi-session priority management
chat = nxrt.load("llama.onnx", priority="realtime", options={"memory.gpu_budget_mb": "10000"})
image = nxrt.load("sdxl.onnx", priority="background", options={"memory.gpu_budget_mb": "6000"})
```

### 31.9 Design Choices

| Choice | Decision | Rationale |
|--------|----------|----------|
| Global resource broker | **Yes** | Sessions must coordinate, not fight. |
| Priority-based preemption | **Yes** | Realtime chat > background image gen. |
| IR sharding annotations | **Yes** | TP/PP need IR-level tensor distribution info. |
| Sharding as optimizer pass | **Yes** | Clean separation: user says "TP=4", pass does the work. |
| NCCL for multi-GPU comm | **Yes** | Industry standard. NVLink bandwidth critical for TP. |
| Automatic TP degree selection | **Stretch goal** | Start with explicit, later auto-detect optimal degree. |
| DP handled by scheduler | **Yes** | No IR change needed — just replicate and split batches. |
| Multi-node support | **Phase 5+** | Focus on single-node multi-GPU first. |

---

## 32. Unified Memory Budget & GenAI Coordination

### 32.1 Problem

Two independent allocators (runtime for weights/activations, GenAI for KV cache)
don't know each other's usage → combined they OOM.

**Design: KV cache stays in GenAI layer (owns the semantics), but both layers
share a single memory budget with pressure-based coordination.**

### 32.2 Shared Budget Interface

```rust
/// Shared device memory budget. Runtime creates it; consumers register partitions.
pub struct DeviceMemoryBudget {
    device: DeviceId,
    total_bytes: usize,
    partitions: Vec<MemoryPartition>,
}

pub struct MemoryPartition {
    name: String,                        // "weights", "kv_cache", "activations", "scratch"
    used: Arc<AtomicUsize>,              // owner updates this
    min_bytes: usize,                    // guaranteed minimum (never shrink below)
    max_bytes: usize,                    // hard ceiling
    priority: EvictionPriority,          // who shrinks first under pressure
    on_pressure: Box<dyn Fn(PressureEvent) -> ShrinkResult + Send>,
}

pub enum PressureEvent {
    /// Another partition needs memory. Can you release some?
    ShrinkRequest { needed_bytes: usize, urgency: Urgency },
    /// You're approaching max. Proactive eviction recommended.
    ApproachingMax { remaining_bytes: usize },
}

pub enum ShrinkResult {
    /// Released this many bytes.
    Released(usize),
    /// Can't release right now (in use).
    Unavailable,
    /// Released partially.
    Partial(usize),
}

pub enum Urgency {
    /// Best-effort, async is fine.
    Low,
    /// Need it soon (within next few ms).
    Medium,
    /// Blocking on this. Synchronous eviction required.
    High,
}
```

### 32.3 Adaptive Budget Strategy (Smart)

Instead of static percentages, observe actual serving patterns and adapt:

```rust
pub struct AdaptiveBudgetManager {
    /// Historical usage samples per partition.
    history: HashMap<String, UsageHistory>,
    /// Current allocation.
    current: BudgetAllocation,
    /// Rebalance interval.
    rebalance_every: Duration,
}

struct UsageHistory {
    /// Rolling window of peak usage per time interval.
    peaks: VecDeque<(Instant, usize)>,
    /// P95 usage over recent window.
    p95_usage: usize,
    /// Growth rate (bytes/sec) — predicts future need.
    growth_rate: f64,
    /// Utilization ratio (actual_used / allocated).
    utilization: f64,
}

impl AdaptiveBudgetManager {
    /// Called periodically. Observes usage patterns, rebalances.
    fn rebalance(&mut self) {
        for (name, history) in &self.history {
            let partition = self.current.get_mut(name);

            // Under-utilized? Shrink soft limit, give to others.
            if history.utilization < 0.5 {
                let excess = partition.allocated - history.p95_usage;
                partition.soft_limit -= excess * 50 / 100;  // give back 50% of excess
            }

            // Growing fast? Proactively expand before it hits pressure.
            if history.growth_rate > 0.0 {
                let predicted_need = history.p95_usage + 
                    (history.growth_rate * self.rebalance_every.as_secs_f64()) as usize;
                partition.soft_limit = max(partition.soft_limit, predicted_need);
            }
        }
    }
}
```

**Serving pattern awareness:**

```rust
/// The budget manager understands common serving phases:
pub enum ServingPhase {
    /// Prefill: activation-heavy, KV growing fast, weights static.
    /// Strategy: give activations more room, allow KV to grow.
    Prefill,
    /// Decode: activation small (single token), KV stable/slow-growing, weights hot.
    /// Strategy: maximize weights on GPU for decode throughput.
    Decode,
    /// Idle: no active requests.
    /// Strategy: preload weights for next request, keep hot KV pages.
    Idle,
    /// Batch transition: new batch arriving, old batch finishing.
    /// Strategy: prepare for burst of KV allocation.
    BatchTransition,
}

impl AdaptiveBudgetManager {
    /// Phase detection from runtime signals.
    fn detect_phase(&self) -> ServingPhase {
        let kv_growth = self.history["kv_cache"].growth_rate;
        let activation_usage = self.history["activations"].utilization;
        let active_requests = self.request_count();

        if active_requests == 0 { return ServingPhase::Idle; }
        if activation_usage > 0.7 && kv_growth > 1_000_000.0 {
            return ServingPhase::Prefill;  // high activation + KV growing fast
        }
        ServingPhase::Decode  // steady state
    }

    /// Adjust budget based on detected phase.
    fn apply_phase_policy(&mut self, phase: ServingPhase) {
        match phase {
            ServingPhase::Prefill => {
                // Activations need more space. Temporarily shrink weight budget.
                // KV is growing — don't evict KV pages right now.
                self.shift_budget("weights", "activations", /* up to */ 20_pct);
            }
            ServingPhase::Decode => {
                // Activations tiny (one token). Maximize weights on GPU.
                // KV stable — can reclaim from activation budget.
                self.shift_budget("activations", "weights", /* up to */ 15_pct);
            }
            ServingPhase::Idle => {
                // Prefetch weights for likely next request.
                // Keep recently-used KV pages (might get continued).
                self.prefetch_hot_weights();
            }
            ServingPhase::BatchTransition => {
                // Burst KV allocation coming. Pre-evict low-priority KV pages.
                self.preemptive_kv_eviction();
            }
        }
    }
}
```

### 32.4 Coordination Protocol

```
Example: Decode → Prefill transition (new long-context request arrives)

1. GenAI scheduler: new request, context_len=32K, estimated KV = 2GB
2. GenAI → budget.reserve("kv_cache", 2GB, Urgency::Medium)
3. Budget manager: current phase=Decode, KV has 500MB free, need 1.5GB more
4. Budget → detect phase shift to Prefill
5. Budget → weights.on_pressure(ShrinkRequest { 1GB, Medium })
   → runtime offloads 1GB of offloadable weights to CPU
6. Budget → activations.on_pressure(ShrinkRequest { 500MB, Medium })
   → runtime shrinks activation arena (no active prefill yet)
7. Budget grants 2GB to kv_cache partition
8. GenAI allocates KV pages on GPU
9. Prefill runs (activations now need space too)
10. Budget: activations partition requests from kv_cache slack
    → KV growth slows during prefill compute → grants temporarily
```

### 32.5 GenAI Integration Point

```rust
// In onnx-genai-engine:
impl GenAiEngine {
    pub fn new(session: &InferenceSession) -> Self {
        let budget = session.device_memory_budget();

        // Register KV cache as a budget partition
        budget.register(MemoryPartition {
            name: "kv_cache".into(),
            used: self.kv_cache.usage_counter(),
            min_bytes: 256 * MB,  // at least 256MB guaranteed
            max_bytes: budget.total() * 60 / 100,
            priority: EvictionPriority::Medium,  // evict before weights, after scratch
            on_pressure: Box::new(|event| {
                match event {
                    PressureEvent::ShrinkRequest { needed_bytes, .. } => {
                        let evicted = self.kv_cache.evict_lru(needed_bytes);
                        ShrinkResult::Released(evicted)
                    }
                    _ => ShrinkResult::Unavailable,
                }
            }),
        });
    }
}
```

### 32.6 User API

```python
# Zero-config: adaptive budget (default)
session = nxrt.load("model.onnx")

# Hint expected workload (helps initial budget planning)
session = nxrt.load("model.onnx", options={
    "memory.expected_context_len": "32768",  # expect 32K context
    "memory.expected_batch_size": "8",       # expect 8 concurrent requests
})

# Override KV cache ceiling
session = nxrt.load("model.onnx", options={
    "memory.kv_cache_max_gb": "6",
})

# Monitor budget live (for dashboards)
budget = session.memory_budget()
print(budget.partitions())  
# {'weights': {used: 4.2GB, limit: 6GB}, 'kv_cache': {used: 2.1GB, limit: 5GB}, ...}
```

### 32.7 Design Choices

| Choice | Decision | Rationale |
|--------|----------|----------|
| KV cache stays in GenAI layer | **Yes** | Semantic ownership (page table, prefix tree, per-request lifecycle). |
| Unified budget interface | **Yes** | Single source of truth for device memory. No independent OOM. |
| Adaptive (not static) | **Yes (default)** | Serving patterns change dynamically. Static splits waste memory. |
| Phase-aware rebalancing | **Yes** | Prefill vs decode have fundamentally different memory profiles. |
| Pressure protocol (not centralized control) | **Yes** | Each partition knows how to shrink itself best. Budget just coordinates. |
| Workload hints | **Optional** | Helps initial allocation but adaptive catches up quickly. |
| Budget observable (API) | **Yes** | Users/dashboards can monitor partition usage in real time. |

---

## 33. Unified Paged Memory (All Tiers)

### 33.1 Design

All memory (VRAM, unified memory, RAM, SSD) is managed as a single paged virtual address space.
Same page abstraction everywhere. Pages migrate transparently between tiers.

```rust
pub struct UnifiedPageTable {
    pages: Vec<PageEntry>,
    page_size: usize,  // 2MB default (matches CUDA huge page / OS huge page)
}

pub struct PageEntry {
    id: PageId,
    location: PageLocation,
    content: PageContent,
    last_access: Instant,
    access_count: u32,
    dirty: bool,
    migration_state: MigrationState,
}

pub enum PageLocation {
    Vram { device: u32, offset: usize },
    UnifiedMemory { offset: usize },       // Apple Silicon / iGPU shared memory
    Ram { host_ptr: *mut u8 },
    Ssd { file: FileId, offset: u64 },     // mmap'd NVMe
    InFlight { from: Tier, to: Tier },     // migration in progress
}

pub enum PageContent {
    Weight { tensor_id: TensorId, shard_index: usize },
    KvCache { layer: u32, block_id: u32 },
    Activation { node_id: NodeId },
    Scratch,
}

pub enum Tier {
    Vram,
    UnifiedMemory,
    Ram,
    Ssd,
}
```

### 33.2 Tier-Specific Behavior

| Tier | Bandwidth | Latency | Migration Cost | Notes |
|------|-----------|---------|----------------|-------|
| VRAM (HBM) | ~2 TB/s | ~ns | — | Fastest. Primary compute tier. |
| Unified Memory | ~200 GB/s | ~ns | Zero (remap only) | Apple Silicon, iGPU. No copy needed. |
| RAM (DDR5) | ~50 GB/s | ~100ns | DMA H2D/D2H | Standard offload target. |
| SSD (NVMe) | ~7 GB/s | ~10µs | Async page-in/out | For 405B on laptop. |

**Unified Memory special case:** On Apple Silicon / iGPUs with shared memory,
no DMA copy needed. "Migration" is just changing memory protection flags:

```rust
impl PageMigrator {
    fn migrate(&self, page: PageId, target: Tier) -> Result<()> {
        match (self.current_tier(page), target) {
            (Tier::Ram, Tier::Vram) => self.dma_h2d_async(page),
            (Tier::Vram, Tier::Ram) => self.dma_d2h_async(page),
            (Tier::Ram, Tier::UnifiedMemory) | (Tier::UnifiedMemory, Tier::Ram) => {
                self.remap_permissions(page);  // zero-copy, just access flags
            }
            (Tier::Ssd, target) => self.async_page_in(page, target),
            (source, Tier::Ssd) => self.async_page_out(page, source),
            _ => Ok(())
        }
    }
}
```

### 33.3 Per-Tier Usage Limits

Users can cap each tier independently:

```python
session = nxrt.load("model.onnx", options={
    "memory.vram_limit_gb": "12",       # max 12GB VRAM (leave room for other apps)
    "memory.ram_limit_gb": "32",        # max 32GB RAM
    "memory.ssd_limit_gb": "100",       # max 100GB SSD
    "memory.ssd_path": "/mnt/nvme/ort_cache",  # where to store pages on disk
    "memory.unified_memory_limit_gb": "24",    # for Apple Silicon
})
```

```rust
pub struct TierLimits {
    pub vram_bytes: Option<usize>,
    pub unified_memory_bytes: Option<usize>,
    pub ram_bytes: Option<usize>,
    pub ssd_bytes: Option<usize>,
    pub ssd_path: Option<PathBuf>,
}
```

---

## 34. Model Resource Estimation

### 34.1 Purpose

Before loading a model, users need to know: "Can my hardware run this? How fast?"

### 34.2 CLI

```bash
$ nxrt inspect model.onnx

┌─────────────────────────────────────────────────────────┐
│  Model: Llama-3.1-70B-Instruct (FP16)                   │
├─────────────────────────────────────────────────────────┤
│  Weights:           140.0 GB (FP16)                      │
│  Opset:             22                                   │
│  Nodes:             4,821                                │
│  Parameters:        70.6B                                │
├─────────────────────────────────────────────────────────┤
│  MEMORY ESTIMATE (context_len=4096, batch=1)             │
│  ──────────────────────────────────────────────────── │
│  Weights:             140.0 GB                           │
│  KV Cache (peak):      5.2 GB                           │
│  Activations (peak):   1.8 GB                           │
│  Scratch:              0.3 GB                            │
│  TOTAL:              147.3 GB                            │
├─────────────────────────────────────────────────────────┤
│  YOUR HARDWARE                                           │
│  ──────────────────────────────────────────────────── │
│  GPU: RTX 4090 (24 GB)                                   │
│  RAM: 64 GB                                              │
│  NVMe: 2 TB                                              │
├─────────────────────────────────────────────────────────┤
│  PLACEMENT PLAN                                          │
│  ──────────────────────────────────────────────────── │
│  VRAM:  24.0 GB → layers 0-12 weights + KV + activations │
│  RAM:   50.2 GB → layers 13-79 weights                   │
│  SSD:    0.0 GB → (not needed)                           │
│                                                          │
│  ⚡ Estimated performance:                                │
│    Prefill (4096 tokens):  ~3.2 sec                      │
│    Decode (per token):     ~180 ms                       │
│    Tokens/sec:             ~5.5 tok/s                    │
│                                                          │
│  💡 Tip: Use INT4 quantization for ~4x memory reduction   │
│     → would fit entirely in VRAM (~35 GB)                │
└─────────────────────────────────────────────────────────┘

$ nxrt inspect model.onnx --context-len 32768 --batch 8 --json
# Outputs structured JSON for programmatic use
```

### 34.3 Python API

```python
import nxrt

# Estimate without loading (fast — only reads metadata + weight sizes)
estimate = nxrt.estimate("model.onnx", context_len=4096, batch_size=1)
print(estimate)
# ResourceEstimate(
#   weights_gb=140.0, kv_cache_gb=5.2, activations_gb=1.8,
#   total_gb=147.3, fits_in_vram=False,
#   placement_plan=PlacementPlan(vram=24GB, ram=50GB, ssd=0),
#   estimated_tok_per_sec=5.5,
#   suggestions=["Use INT4 quantization to fit in VRAM"]
# )

# Check if a specific config is feasible
nxrt.estimate("model.onnx", context_len=128000, batch_size=32,
             hardware={"vram_gb": 80, "ram_gb": 256})
```

### 34.4 Estimation Logic

```rust
pub struct ResourceEstimate {
    pub weights_bytes: usize,
    pub kv_cache_bytes: usize,       // f(num_layers, num_heads, head_dim, context_len, batch)
    pub activation_peak_bytes: usize, // f(hidden_size, context_len, batch) — max across layers
    pub scratch_bytes: usize,         // workspace for kernels (e.g. cuBLAS)
    pub total_bytes: usize,
    pub placement_plan: PlacementPlan,
    pub estimated_latency: LatencyEstimate,
    pub suggestions: Vec<String>,
}

impl ResourceEstimator {
    pub fn estimate(model_path: &Path, params: &EstimateParams) -> Result<ResourceEstimate> {
        // Fast path: read only model metadata + weight tensor shapes
        // No weight loading, no graph optimization
        let meta = quick_metadata_scan(model_path)?;

        let weights = meta.total_weight_bytes();
        let kv = estimate_kv_cache(
            meta.num_layers, meta.num_kv_heads, meta.head_dim,
            params.context_len, params.batch_size, meta.kv_dtype,
        );
        let activations = estimate_peak_activation(
            meta.hidden_size, params.context_len, params.batch_size,
        );

        // Generate placement plan against detected/specified hardware
        let plan = plan_placement(weights, kv, activations, &params.hardware);

        // Estimate latency from cost model (bandwidth + FLOPs)
        let latency = estimate_latency(&plan, &meta, params);

        // Generate actionable suggestions
        let suggestions = generate_suggestions(&plan, &meta);

        Ok(ResourceEstimate { weights, kv, activations, .. })
    }
}
```

---

## 35. Error Recovery & Debug Experience

### 35.1 Philosophy

**Errors should feel like a helpful colleague explaining what went wrong,
not a stack trace from hell.**

Every error message answers three questions:
1. **What** went wrong (specific, not generic)
2. **Why** it happened (root cause, not symptom)
3. **How** to fix it (actionable suggestion)

### 35.2 Error Types

```rust
pub struct RuntimeError {
    pub kind: ErrorKind,
    pub message: String,
    pub context: ErrorContext,
    pub suggestion: Option<String>,
    pub docs_url: Option<String>,
}

pub struct ErrorContext {
    /// Which op/node triggered this.
    pub node_name: Option<String>,
    pub op_type: Option<String>,
    /// Input shapes at the point of failure.
    pub input_shapes: Vec<(String, Vec<usize>)>,
    /// What was expected.
    pub expected: Option<String>,
    /// What was received.
    pub actual: Option<String>,
}

pub enum ErrorKind {
    // --- Recoverable (runtime continues) ---
    /// Shape mismatch on input.
    ShapeMismatch,
    /// Data type mismatch.
    DtypeMismatch,
    /// Timeout on inference (configurable deadline).
    Timeout,
    /// Model file not found or corrupted.
    ModelLoad,
    /// Unknown op (no kernel registered).
    UnsupportedOp,
    /// Resource exhaustion (handled by offloading, but reported).
    ResourcePressure,

    // --- Fatal (session must be recreated) ---
    /// CUDA context corrupted (sticky error from kernel crash).
    DeviceContextCorrupt,
    /// Hardware failure (ECC, device lost).
    HardwareFailure,
    /// Internal bug (invariant violated). Always include repro info.
    InternalError,
}
```

### 35.3 Example Error Messages

```
❌ Shape mismatch on input "input_ids"

  Expected: [batch_size, sequence_length] with dtype int64
  Got:      [1, 128, 768] with dtype float32
            ^^^^^^^^^^^
            3 dimensions, expected 2

  This looks like you're passing hidden states instead of token IDs.
  input_ids should be integer token IDs from your tokenizer.

  Docs: https://nxrt.dev/errors/shape-mismatch
```

```
❌ Unsupported operator: "com.microsoft.NewFancyOp" (version 2)

  This model uses a Microsoft contrib op that our runtime doesn't implement yet.

  Options:
  1. Use backend="ort" to run this model on upstream ORT
  2. Register a custom op: session.register_custom_ops("path/to/plugin.so")
  3. Open an issue: https://github.com/justinchuby/onnx-genai/issues/new

  Node: "model.layers.5.mlp.NewFancyOp"
  Required by: 3 nodes in the graph
```

```
⚠️  GPU memory pressure (using 23.1/24.0 GB)

  Offloading 2.3 GB of weights from GPU to CPU RAM.
  Performance may decrease ~15% for affected layers (13-24).

  To avoid this:
  • Use INT4 quantization: nxrt quantize model.onnx --bits 4
  • Reduce context length (current: 32768)
  • Set memory.gpu_budget_mb to reserve space: options={"memory.gpu_budget_mb": "20000"}
```

### 35.4 CUDA Error Recovery

```rust
impl InferenceSession {
    fn run_with_recovery(&mut self, inputs: &[(&str, &Tensor)]) -> Result<Vec<Tensor>> {
        match self.run_inner(inputs) {
            Ok(result) => Ok(result),
            Err(e) if e.is_cuda_context_corrupt() => {
                // CUDA errors are sticky. Must reset context.
                tracing::error!("CUDA context corrupt. Recovering...");
                self.reset_cuda_context()?;
                // Retry once. If it fails again, return fatal error.
                self.run_inner(inputs)
            }
            Err(e) => Err(e)
        }
    }

    fn reset_cuda_context(&mut self) -> Result<()> {
        // 1. Destroy current CUDA context
        // 2. Create fresh context
        // 3. Reload weights from checkpoint (weights are on CPU/SSD anyway)
        // 4. Rebuild execution plan
        // User sees: one failed request, next request works.
        Ok(())
    }
}
```

### 35.5 Hardware Degradation

```rust
pub enum HardwareEvent {
    ThermalThrottle { device: DeviceId, temp_celsius: u32 },
    EccError { device: DeviceId, count: u32 },
    NvLinkFailure { device_a: DeviceId, device_b: DeviceId },
    DeviceLost { device: DeviceId },
}

impl ResourceBroker {
    fn handle_hardware_event(&mut self, event: HardwareEvent) {
        match event {
            HardwareEvent::DeviceLost { device } => {
                // 1. Mark device unavailable
                // 2. Replan all sessions that used this device
                // 3. Continue serving on remaining devices (degraded)
                // 4. Alert operator
                // NEVER crash the server.
            }
            HardwareEvent::ThermalThrottle { device, temp } => {
                // Reduce load on this device, shift work to cooler devices
                self.reduce_device_load(device, 50 /* percent */);
            }
            _ => {}
        }
    }
}
```

---

## 36. Determinism & Reproducibility

```python
session = nxrt.load("model.onnx", options={
    "execution.deterministic": "true",
})
```

| Source of Non-Determinism | Deterministic Path | Cost |
|---|---|---|
| cuBLAS workspace algorithms | Force deterministic algorithm | ~5-10% slower |
| Atomic reductions (LayerNorm, Softmax) | Sequential reduction | ~10-20% slower |
| FlashAttention warp scheduling | Use deterministic FA variant | ~5% slower |
| Thread scheduling (CPU) | Pin to fixed thread order | Minimal |

When `deterministic=true`: same input → **bit-exact** same output, every time.
Default is `false` (performance over reproducibility).

---

## 37. VMap (Auto-Batching)

### 37.1 Problem

Many ONNX models are exported with `batch_size=1`. Users want to run batch=8/16/32
without re-exporting. JAX's `vmap` solves this at the framework level.

### 37.2 Design: Graph-Level VMap Pass

```rust
/// VMap: vectorize a batch_size=1 graph to arbitrary batch size.
/// Analogous to JAX's vmap — maps a function over a batch dimension.
pub struct VMapPass {
    /// Which inputs get the new batch dimension.
    batched_inputs: Vec<String>,
    /// Target batch size (or dynamic).
    batch_size: BatchSize,
    /// Which axis is the batch axis (default: 0).
    batch_axis: usize,
}

pub enum BatchSize {
    Static(usize),   // compile-time known (enables more optimization)
    Dynamic,         // symbolic, determined at runtime
}

impl OptimizerPass for VMapPass {
    fn run(&self, graph: &mut Graph) -> Result<()> {
        // For each node, determine how to vectorize:
        for node in graph.nodes_topo() {
            match self.vectorize_rule(node) {
                VMapRule::Broadcast => {
                    // Op naturally broadcasts (Add, Mul, etc.)
                    // Just update shape annotations
                    self.update_shapes(node, self.batch_size);
                }
                VMapRule::BatchMatMul => {
                    // MatMul [B,M,K] x [K,N] → batched GEMM
                    self.promote_to_batched_gemm(node);
                }
                VMapRule::Replicate => {
                    // Non-batchable op (rare): replicate for each batch element
                    self.replicate_op(node, self.batch_size);
                }
                VMapRule::Reshape => {
                    // Reshape/Squeeze/Unsqueeze: adjust shape constants
                    self.adjust_shape_constants(node);
                }
            }
        }
        Ok(())
    }
}

enum VMapRule {
    /// Op broadcasts naturally over batch dim (elementwise, reduction with keepdim).
    Broadcast,
    /// MatMul promotes to batched GEMM.
    BatchMatMul,
    /// Can't vectorize. Replicate the op per batch element (fallback).
    Replicate,
    /// Shape manipulation: adjust constants to account for batch dim.
    Reshape,
}
```

### 37.3 How It Works (Example)

```
Original graph (batch=1):
  input: [1, 128, 4096]
  MatMul: [1, 128, 4096] @ [4096, 4096] → [1, 128, 4096]
  LayerNorm: [1, 128, 4096] → [1, 128, 4096]

After VMap(batch_size=8):
  input: [8, 128, 4096]
  MatMul: [8, 128, 4096] @ [4096, 4096] → [8, 128, 4096]  (batched GEMM)
  LayerNorm: [8, 128, 4096] → [8, 128, 4096]  (broadcasts over batch)
```

Most transformer ops naturally support batch dimension. The hard cases:
- **Reshape with hardcoded shapes** → VMap rewrites shape constants
- **Gather with batch-dependent indices** → adjust index offsets
- **Custom ops** → fallback to Replicate (run N times)

### 37.4 User API

```python
# Load batch=1 model, run with batch=8
session = nxrt.load("model.onnx", options={
    "vmap.batch_size": "8",           # static batch
    "vmap.batch_inputs": "input_ids,attention_mask",  # which inputs are batched
})
# Now session.run accepts [8, seq_len] inputs

# Dynamic batching (batch size varies per call)
session = nxrt.load("model.onnx", options={
    "vmap.batch_size": "dynamic",
    "vmap.max_batch": "32",           # pre-compile up to 32
})

# Programmatic (Rust)
let session = InferenceSession::builder()
    .model("model.onnx")
    .vmap(VMapConfig {
        batch_size: BatchSize::Static(8),
        batched_inputs: vec!["input_ids".into(), "attention_mask".into()],
        batch_axis: 0,
    })
    .build()?;
```

### 37.5 Difficulty Assessment

| Op Category | VMap Difficulty | Notes |
|---|---|---|
| Elementwise (Add, Mul, Gelu) | Trivial | Natural broadcast |
| MatMul/Gemm | Easy | Batched GEMM (cuBLAS supports natively) |
| LayerNorm/RMSNorm | Easy | Operates on last dims, batch is first |
| Attention (GQA/MHA) | Easy | Already batch-aware |
| Reshape/Squeeze/Unsqueeze | Medium | Must rewrite shape constants |
| Gather/ScatterND | Medium | Index offsets need adjustment |
| Control flow (If/Loop) | Hard | Must unroll or replicate |
| Custom ops | Hard | Fallback to replicate |

For transformer models: **95%+ of ops vmap trivially.** Main complexity is
shape-manipulation ops with hardcoded constants.

---

## 38. Prefill/Decode Execution Mode

### 38.1 Decision: Unified Dual-Track (not split sessions)

Same session, two execution tracks optimized for different compute profiles:

```rust
pub struct DualTrackScheduler {
    prefill_track: ExecutionTrack,  // compute-bound, large GEMM
    decode_track: ExecutionTrack,   // memory-bound, CUDA graph replay
    shared: Arc<SharedState>,       // weights + KV cache (zero-copy between tracks)
}

pub enum PrefillDecodeMode {
    /// Default: unified session, chunked prefill interleaved with decode.
    Unified { prefill_chunk_size: usize },
    /// Multi-node: separate devices for prefill vs decode. KV shipped over network.
    Disaggregated { prefill_devices: Vec<DeviceId>, decode_devices: Vec<DeviceId> },
}
```

Chunked prefill prevents head-of-line blocking:
```
Time: [prefill chunk 512 tok][decode batch][prefill chunk 512][decode batch]...
→ decode latency stays bounded even during long prefills
```

### 38.2 Multi-Session Scheduling

When multiple sessions share the same runtime, the **ResourceBroker** arbitrates:

```rust
impl ResourceBroker {
    /// Sessions request execution slots. Broker decides who runs when.
    fn schedule_next(&mut self) -> ScheduleDecision {
        // Priority queue: realtime decode > foreground prefill > background
        // Within same priority: round-robin fairness
        // Preemption: realtime can interrupt background mid-chunk

        let candidates: Vec<_> = self.sessions
            .iter()
            .filter(|s| s.has_pending_work())
            .sorted_by_key(|s| s.priority)
            .collect();

        if let Some(realtime) = candidates.iter().find(|s| s.priority == Realtime && s.needs_decode()) {
            // Realtime decode always goes first (latency-sensitive)
            return ScheduleDecision::RunDecode(realtime.id);
        }

        // Fair scheduling among same-priority sessions
        self.round_robin_among(candidates)
    }
}

pub struct SessionResourceRequest {
    session_id: SessionId,
    /// What this session wants to do next.
    work_type: WorkType,
    /// How much GPU time it needs (estimated ms).
    estimated_duration_ms: f64,
    /// Memory it will allocate during this step.
    memory_delta: isize,
}

pub enum WorkType {
    /// Prefill chunk (compute-heavy, known duration).
    Prefill { tokens: usize },
    /// Decode step (fast, latency-critical).
    Decode { batch_size: usize },
    /// Background (optimization, weight loading).
    Background,
}
```

**Session requests resources via the broker:**
```rust
impl InferenceSession {
    /// Before running, session checks in with broker.
    fn request_execution_slot(&self) -> Result<ExecutionGrant> {
        let request = SessionResourceRequest {
            session_id: self.id,
            work_type: self.next_work_type(),
            estimated_duration_ms: self.estimate_next_step_ms(),
            memory_delta: self.estimate_memory_delta(),
        };

        // Broker may:
        // - Grant immediately (resources available)
        // - Queue (higher priority session running)
        // - Shrink this session's budget first, then grant
        self.broker.request(request)
    }
}

pub enum ExecutionGrant {
    /// Go ahead. You have this much time and memory.
    Granted { time_budget_ms: f64, memory_budget: usize },
    /// Wait. Another session has priority.
    Queued { position: usize, estimated_wait_ms: f64 },
    /// You need to shrink first. Release this much memory.
    ShrinkFirst { release_bytes: usize },
}
```

---

## 39. Streaming (Runtime ↔ GenAI Integration)

### 39.1 Streaming is GenAI-Layer Concern

The runtime itself doesn't "stream tokens" — it runs one forward pass and returns
a tensor. Streaming is the GenAI engine calling runtime repeatedly in a decode loop.

But the runtime needs to support **efficient repeated execution** that enables streaming:

```rust
/// Runtime provides: fast repeated execution with minimal overhead.
impl InferenceSession {
    /// Hot path for decode loop. Reuses buffers, replays CUDA graph.
    /// This IS the streaming primitive — GenAI calls it once per token.
    fn run_decode_step(&mut self, inputs: &[(&str, &Tensor)]) -> Result<Vec<Tensor>>;
}

/// GenAI layer wraps this in streaming:
impl GenAiEngine {
    pub fn generate_stream(
        &mut self,
        prompt_tokens: &[u32],
        callback: impl FnMut(StreamEvent) -> ControlFlow,
    ) -> Result<()> {
        // Prefill
        self.prefill(prompt_tokens)?;
        callback(StreamEvent::PrefillDone { tokens: prompt_tokens.len() });

        // Decode loop (streaming)
        loop {
            let logits = self.session.run_decode_step(&self.decode_inputs())?;
            let token = self.sample(&logits);
            if token == self.eos_token { break; }

            match callback(StreamEvent::Token { token, text: self.decode(token) }) {
                ControlFlow::Continue => {}
                ControlFlow::Break => break,  // user cancelled
            }
        }
        Ok(())
    }
}
```

### 39.2 What Runtime Provides for Streaming Performance

| Feature | Purpose |
|---------|--------|
| Buffer reuse (§20.3) | No malloc per decode step |
| CUDA graph replay | Near-zero launch overhead per step |
| `run_decode_step()` fast path | Skip input validation on hot path |
| Async KV cache append | Overlap KV write with next step's compute |

---

## 40. Concurrency Model & Thread Safety

### 40.1 Design

`session.run()` takes `&mut self` (single-threaded per session instance).
For concurrent serving: lightweight clones share weights, independent buffers.

```rust
impl InferenceSession {
    /// Lightweight clone: Arc-shared weights, independent activation/scratch buffers.
    /// Use for multi-threaded serving (one clone per worker thread).
    pub fn clone_for_concurrency(&self) -> Self {
        Self {
            weights: Arc::clone(&self.weights),        // shared, read-only
            execution_plan: Arc::clone(&self.plan),    // shared, immutable
            activation_buffers: ActivationBuffers::new_independent(),  // per-clone
            scratch: ScratchSpace::new(),              // per-clone
            output_cache: OutputBufferCache::new(),    // per-clone
            profiling: self.profiling.clone(),         // shared collector
        }
    }
}

/// Serving pattern:
let base_session = InferenceSession::load("model.onnx")?;
let pool: Vec<_> = (0..num_workers)
    .map(|_| base_session.clone_for_concurrency())
    .collect();

// Each worker thread owns its clone. No locks on the hot path.
pool.par_iter_mut().zip(requests).for_each(|(session, req)| {
    session.run(&req.inputs).unwrap();
});
```

### 40.2 What's Shared vs Independent

| Component | Shared (Arc) | Independent (per-clone) |
|-----------|:---:|:---:|
| Model weights | ✅ | |
| Execution plan | ✅ | |
| KV cache | | ✅ (per-request) |
| Activation buffers | | ✅ |
| Scratch space | | ✅ |
| Output buffer cache | | ✅ |
| Profiling collector | ✅ | |
| Resource claim (broker) | ✅ | |

---

## 41. Compilation Cache (Cold Start)

### 41.1 Problem

Cold start: load model → optimize → plan placement → compile kernels → first run.
Serverless: every cold start costs 500ms-5s. Unacceptable.

### 41.2 Persistent Cache

```rust
pub struct CompilationCache {
    cache_dir: PathBuf,  // default: ~/.cache/nxrt/
}

impl CompilationCache {
    /// Cache key = hash(model_bytes + device_profile + options + runtime_version)
    fn cache_key(model: &[u8], device: &DeviceProfile, options: &SessionOptions) -> CacheKey {
        let mut hasher = blake3::Hasher::new();
        hasher.update(model);
        hasher.update(&device.fingerprint());
        hasher.update(&options.fingerprint());
        hasher.update(env!("CARGO_PKG_VERSION").as_bytes());
        CacheKey(hasher.finalize())
    }

    /// Save compiled plan after first load.
    fn save(&self, key: CacheKey, plan: &CompiledPlan) -> Result<()>;

    /// Load cached plan. Returns None if cache miss or invalidated.
    fn load(&self, key: CacheKey) -> Option<CompiledPlan>;
}

pub struct CompiledPlan {
    pub optimized_graph: SerializedGraph,
    pub placement: PlacementPlan,
    pub memory_offsets: AotMemoryPlan,
    pub kernel_selections: HashMap<NodeId, KernelId>,
    pub cuda_graph_regions: Vec<CudaGraphRegion>,
}
```

**Performance:**
```
First load:    load(200ms) → optimize(300ms) → plan(100ms) → compile(200ms) = ~800ms
Cached load:   load_plan(50ms) → mmap_weights(10ms) → warm_kernels(50ms) = ~110ms
                                                                    7x faster
```

> **Relationship to the on-disk `EPContext` node (§57).** This cache is our
> *internal, host-local* cold-start artifact keyed on `hash(model + device +
> options + runtime_version)` — it caches the *whole compiled plan* and is not
> portable across machines or ORT-compatible tools. ORT's `EPContext` contrib
> node (§57) is the *portable, in-graph* form of a single EP's compiled context:
> it travels inside the `*_ctx.onnx` model and is consumed by any ORT-compatible
> runtime. The two are complementary — when an EP's `save_context()` produces an
> `EpContext`, its `data` blob can be stored both here (fast local reload) and,
> when `ep.context_enable` is set, serialized into an `EPContext` node for
> portable distribution.

### 41.3 CLI

```bash
# Pre-compile (e.g. in Docker build step)
nxrt compile model.onnx --output model.nxrtplan --device gpu

# Load pre-compiled (instant)
nxrt run model.nxrtplan --input data.npz

# Cache management
nxrt cache list
nxrt cache clear
nxrt cache clear --older-than 30d
```

---

## 42. Model Security

```rust
pub enum ModelTrustLevel {
    /// Fully trusted. Allow custom ops, dlopen, native EPs.
    Trusted,
    /// Internal. Allow known custom op domains only.
    Internal { allowed_domains: HashSet<String> },
    /// Untrusted (user-uploaded). Sandboxed execution.
    Untrusted,
}

pub struct SecurityPolicy {
    trust_level: ModelTrustLevel,
    max_model_bytes: usize,         // DoS prevention
    max_tensor_bytes: usize,        // OOM bomb prevention
    max_nodes: usize,               // graph complexity limit
    allow_external_data: bool,      // can model reference external files?
    allow_custom_ops: bool,         // can model use dlopen'd ops?
}
```

`Untrusted` mode: no `dlopen`, no external data references, tensor size limits,
graph complexity limits. Safe for serving user-uploaded models.

---

## 43. Metrics & Observability

Beyond per-request profiling, production needs aggregated metrics:

```rust
pub struct RuntimeMetrics {
    // Latency
    pub inference_latency_p50_us: f64,
    pub inference_latency_p99_us: f64,
    pub prefill_latency_p50_us: f64,
    pub decode_step_latency_p50_us: f64,

    // Throughput
    pub requests_total: u64,
    pub tokens_generated_total: u64,
    pub tokens_per_second: f64,

    // Resource utilization
    pub gpu_utilization_pct: f64,
    pub gpu_memory_used_bytes: usize,
    pub kv_cache_hit_rate: f64,       // prefix cache effectiveness
    pub buffer_reuse_rate: f64,

    // Health
    pub errors_total: u64,
    pub pressure_events_total: u64,
    pub offload_bytes_total: u64,
    pub recovery_count: u64,          // CUDA context resets
}

impl InferenceSession {
    /// Prometheus-compatible metrics export.
    pub fn metrics(&self) -> RuntimeMetrics;

    /// OpenTelemetry integration.
    pub fn otel_meter(&self) -> &Meter;
}
```

```python
# Python
metrics = session.metrics()
print(f"P99 latency: {metrics.inference_latency_p99_us / 1000:.1f}ms")
print(f"Throughput: {metrics.tokens_per_second:.1f} tok/s")
print(f"KV cache hit rate: {metrics.kv_cache_hit_rate:.1%}")
```

```bash
# Prometheus endpoint (for serving)
nxrt serve model.onnx --metrics-port 9090
# GET http://localhost:9090/metrics → prometheus text format
```

---

## 44. Testing Strategy

```
┌──────────────────────────────────────────────────────┐
│ L1: Unit tests per kernel (inputs → expected output)    │
├──────────────────────────────────────────────────────┤
│ L2: Conformance — top 50 HuggingFace models vs ORT      │
│     allclose(our_output, ort_output, atol=1e-5)         │
├──────────────────────────────────────────────────────┤
│ L3: Differential fuzzing (random shapes + inputs)       │
├──────────────────────────────────────────────────────┤
│ L4: Stress — 10K concurrent requests, memory leaks      │
├──────────────────────────────────────────────────────┤
│ L5: Chaos — kill GPU, corrupt input, disk full          │
├──────────────────────────────────────────────────────┤
│ L6: Perf regression CI (latency/throughput benchmarks)  │
└──────────────────────────────────────────────────────┘
```

---

## 45. Design Choices (Comprehensive)

| # | Choice | Decision | Rationale |
|---|--------|----------|----------|
| 1 | Unified paged memory | **Yes** | VRAM/Unified/RAM/SSD as single page table. Pages migrate transparently. |
| 2 | Per-tier user limits | **Yes** | Users can cap each tier independently. |
| 3 | VMap (auto-batching) | **Yes** | Graph-level vectorization pass. batch=1 model → batch=N without re-export. |
| 4 | Prefill/Decode split | **No (unified dual-track)** | Same session, two exec tracks. Disaggregated only for multi-node. |
| 5 | Multi-session broker scheduling | **Yes** | Sessions request execution grants. Broker arbitrates by priority + fairness. |
| 6 | Model resource estimation | **Yes (CLI + API)** | `nxrt inspect` and `nxrt.estimate()` before loading. |
| 7 | Streaming at runtime level | **No** | Runtime provides efficient repeated execution. Streaming is GenAI concern. |
| 8 | Error message quality | **Yes (first-class)** | Every error: what/why/how-to-fix + docs link. |
| 9 | CUDA error recovery | **Yes** | Context reset + session rebuild. One bad request doesn't kill server. |
| 10 | Deterministic mode | **Yes (opt-in)** | `deterministic=true` → bit-exact, 10-20% slower. |
| 11 | Compilation cache | **Yes** | Persistent on-disk cache. 7x faster cold start on cache hit. |
| 12 | Model security levels | **Yes** | Trusted/Internal/Untrusted. Untrusted = no dlopen, size limits. |
| 13 | Prometheus metrics | **Yes** | Aggregated runtime metrics for production monitoring. |
| 14 | Hardware degradation handling | **Yes** | Never crash. Degrade gracefully, replan, alert. |

---

## 46. Execution Trace Log (Debug Mode)

### 46.1 Purpose

When something goes wrong (wrong output, slow inference, mysterious OOM), users need to
**replay what happened** step by step. Not just timing (profiling) but full execution history:
what was computed, what shapes flowed where, what decisions the runtime made.

### 46.2 Design: Structured Execution Journal

```rust
/// Enable with: options={"debug.trace_log": "/tmp/nxrt-trace.jsonl"}
/// or: ORT2_TRACE_LOG=/tmp/nxrt-trace.jsonl
/// or: session.enable_trace_log("/tmp/trace.jsonl")
pub struct TraceLogger {
    output: BufWriter<File>,  // append-only JSONL
    verbosity: TraceVerbosity,
    /// Capture tensor values (expensive! only for small models/debugging)
    capture_values: bool,
}

pub enum TraceVerbosity {
    /// Decisions only: placement, optimization, memory planning.
    /// Small file. Good for "why is it slow?" questions.
    Decisions,
    /// + Per-op execution with shapes and timing.
    /// Medium file. Good for "which op is wrong?" questions.
    Ops,
    /// + Tensor values (inputs/outputs of each op).
    /// HUGE file. Good for "where does the numerical error start?" questions.
    /// Only enable for small models or specific nodes.
    Full,
}
```

### 46.3 What's Logged (JSONL format, one event per line)

```jsonl
{"ts":0,"phase":"load","event":"model_loaded","path":"model.onnx","opset":22,"nodes":4821,"weights_gb":140.0}
{"ts":5,"phase":"optimize","event":"pass_start","pass":"ConstantFolding","nodes_before":4821}
{"ts":12,"phase":"optimize","event":"pass_done","pass":"ConstantFolding","nodes_after":4650,"eliminated":171}
{"ts":13,"phase":"optimize","event":"fusion","pattern":"MatMul+BiasAdd+Gelu→BiasGelu","nodes":["layer.0.mlp.fc1","layer.0.mlp.bias","layer.0.mlp.gelu"],"fused_to":"layer.0.mlp.BiasGelu"}
{"ts":50,"phase":"placement","event":"ilp_solved","solve_time_ms":23,"cost":1847.3,"gpu_nodes":3200,"cpu_nodes":1450}
{"ts":51,"phase":"placement","event":"node_placed","node":"layer.0.attn.gqa","device":"gpu:0","reason":"force_hint"}
{"ts":52,"phase":"placement","event":"node_placed","node":"layer.31.mlp.down","device":"cpu","reason":"gpu_full"}
{"ts":100,"phase":"memory","event":"plan_done","arena_gpu_mb":8192,"arena_cpu_mb":24000,"aliases":342}
{"ts":101,"phase":"memory","event":"weight_placed","tensor":"layer.0.attn.q_proj.weight","size_mb":128,"tier":"vram","priority":1}
{"ts":102,"phase":"memory","event":"weight_placed","tensor":"layer.31.mlp.gate.weight","size_mb":256,"tier":"ram","priority":78}
{"ts":200,"phase":"run","run_id":1,"event":"run_start","inputs":{"input_ids":[1,128]}}
{"ts":201,"phase":"run","event":"kernel_exec","node":"layer.0.attn.q_proj","op":"MatMul","kernel":"cublas_gemm","device":"gpu:0","input_shapes":[[1,128,4096],[4096,4096]],"output_shape":[1,128,4096],"duration_us":85}
{"ts":202,"phase":"run","event":"buffer_reuse","tensor":"layer.0.attn.output","shape":[1,128,4096],"hit":true}
{"ts":203,"phase":"run","event":"prefetch","tensor":"layer.1.attn.q_proj.weight","from":"ram","to":"vram","size_mb":128,"async":true}
{"ts":500,"phase":"run","event":"pressure","partition":"kv_cache","used_mb":5200,"soft_limit_mb":5000,"action":"evict_lru","evicted_pages":4}
{"ts":800,"phase":"run","run_id":1,"event":"run_done","duration_ms":12.3,"output_shapes":{"logits":[1,128,32000]}}
```

**Full verbosity adds tensor values (opt-in per node):**
```jsonl
{"ts":201,"phase":"run","event":"kernel_exec","node":"layer.0.attn.q_proj","values":{"input_0":"tensor:sha256:a3f2...(saved to /tmp/nxrt-tensors/run1_node0_in0.npy)","output_0":"tensor:sha256:b7e1...(saved to /tmp/nxrt-tensors/run1_node0_out0.npy)"}}
```

### 46.4 User API

```python
import nxrt

# Enable via context manager
with nxrt.trace_log("/tmp/debug.jsonl", verbosity="ops") as log:
    session = nxrt.load("model.onnx")
    output = session.run(input_ids=data)
# File ready for inspection

# Or session-level (persistent)
session = nxrt.load("model.onnx", options={
    "debug.trace_log": "/tmp/trace.jsonl",
    "debug.trace_verbosity": "ops",       # decisions | ops | full
    "debug.capture_values": "layer.5.*",  # only capture values for layer 5
})

# Or env var (no code change needed!)
# ORT2_TRACE_LOG=/tmp/trace.jsonl ORT2_TRACE_VERBOSITY=ops python run.py
```

```bash
# CLI: run with tracing
nxrt run model.onnx --input data.npz --trace-log /tmp/trace.jsonl --trace-verbosity ops

# Analyze the trace
nxrt trace analyze /tmp/trace.jsonl

┌───────────────────────────────────────────────────────┐
│  Trace Analysis: /tmp/trace.jsonl                       │
├───────────────────────────────────────────────────────┤
│  Total runs: 1                                          │
│  Total duration: 12.3ms                                 │
│                                                         │
│  Top 5 slowest ops:                                     │
│    1. layer.15.attn.gqa  (GroupQueryAttention)  2.1ms   │
│    2. layer.0.mlp.gate   (MatMul)              1.8ms   │
│    3. layer.0.attn.gqa   (GroupQueryAttention)  1.6ms   │
│    4. layer.1.mlp.gate   (MatMul)              1.5ms   │
│    5. layer.1.attn.gqa   (GroupQueryAttention)  1.4ms   │
│                                                         │
│  Memory events: 3 pressure, 4 pages evicted             │
│  Offloads: 2.3GB weights RAM→GPU prefetched             │
│  Buffer reuse rate: 94%                                 │
│                                                         │
│  ⚠️  Bottleneck: layer.15.attn.gqa is 2x slower than    │
│     other GQA ops. Possible cause: KV pages were        │
│     evicted and re-fetched during this layer.           │
└───────────────────────────────────────────────────────┘

# Compare two traces (e.g. before/after optimization)
nxrt trace diff trace_v1.jsonl trace_v2.jsonl

# Find where numerical divergence starts (full verbosity)
nxrt trace bisect trace_ours.jsonl trace_ort.jsonl
  → "Divergence starts at node layer.5.attn.gqa (output differs by 1e-3)"
```

### 46.5 Trace Replay (Offline Debugging)

The trace log is **replayable** — contains enough info to reconstruct what happened
without re-running the model:

```python
# Load and query trace programmatically
trace = nxrt.TraceLog.load("/tmp/trace.jsonl")

# Query specific events
for event in trace.filter(phase="run", event="pressure"):
    print(f"Pressure at {event.ts}ms: {event.partition} used {event.used_mb}MB")

# Get execution timeline for a specific node
node_events = trace.node_history("layer.15.attn.gqa")
print(node_events)
# [KernelExec(duration=2.1ms, inputs=[[1,128,4096]], prefetch_stall=0.8ms)]

# Export to DataFrame for analysis
df = trace.to_dataframe()
df.groupby("op")["duration_us"].describe()
```

### 46.6 Automatic Diagnosis

The trace analyzer can **automatically identify common problems:**

```rust
pub struct AutoDiagnosis {
    pub issues: Vec<DiagnosedIssue>,
}

pub struct DiagnosedIssue {
    pub severity: Severity,  // info, warning, critical
    pub category: IssueCategory,
    pub description: String,
    pub evidence: Vec<TraceEvent>,
    pub suggestion: String,
}

pub enum IssueCategory {
    /// An op is much slower than expected (outlier detection).
    SlowOp,
    /// Frequent memory pressure events → thrashing.
    MemoryThrashing,
    /// Prefetch not hiding transfer latency (compute < transfer).
    PrefetchStall,
    /// Buffer reallocations (shape instability).
    ShapeInstability,
    /// CUDA graph not captured (shape keeps changing).
    NoCudaGraph,
    /// An optimized kernel existed (e.g. FlashAttention / fused SDPA / fused
    /// GEMM) but the runtime took the generic fallback instead.
    MissedFastPath,
    /// Weight on wrong tier (hot weight on CPU).
    SuboptimalPlacement,
    /// Numerical divergence from reference.
    NumericalDivergence,
}
```

```bash
$ nxrt trace diagnose /tmp/trace.jsonl

⚠️  Memory thrashing detected
   Evidence: 12 pressure events in 500ms, same pages evicted and re-fetched
   Root cause: KV cache and weights competing for last 2GB of VRAM
   Fix: increase memory.vram_limit_gb or reduce context length

⚠️  Missed fast path on 'FusedAttention_3': 'FlashAttention' available but not used
   Evidence: 'FusedAttention_3' could have used the optimized 'FlashAttention'
             kernel, but the runtime ran the 'generic f32 SDPA' fallback instead.
             Reason the fast path was rejected: unsupported dtype fp32.
   Root cause: the runtime evaluated 'FlashAttention' and rejected it because the
               op fell back to a slower generic kernel doing the same math.
   Fix: cast the op's inputs to a supported dtype (fp16/bf16) so FlashAttention engages

⚠️  Prefetch stall on layers 13-20
   Evidence: 0.8ms stall per layer waiting for weight transfer
   Root cause: compute time (0.5ms) < transfer time (1.3ms) for these layers
   Fix: keep these layers' weights on GPU (use placement hint "force")

ℹ️  CUDA graph not captured
   Evidence: shapes changed 3 times in first 10 runs
   Root cause: variable sequence length inputs
   Fix: use padding to stabilize shapes, or set vmap.batch_size=static
```

#### 46.6.1 MissedFastPath emission contract

When an op **has** an optimized kernel (FlashAttention, fused SDPA, a fused/
vendor GEMM, …) but the runtime does **not** take it, that must be *loud* in the
trace — a silent fallback is exactly the kind of invisible perf cliff RULES.md #1
forbids. `MissedFastPath` is **data-conditional**: the kernel/EP that owns the
fast-vs-fallback decision reports the rejection at the point it falls back, and
`AutoDiagnosis` turns that into a WHAT/WHY/HOW issue.

**Trace-arg keys** (stable convention any EP populates on the op's event):

| Key | Const | Meaning |
|-----|-------|---------|
| `optimized_candidate` | `ARG_OPTIMIZED_CANDIDATE` | the optimized kernel that *could* have run (e.g. `"FlashAttention"`) |
| `fastpath_rejected_reason` | `ARG_FASTPATH_REJECTED_REASON` | **why** it was skipped (drives the fix + severity) |
| `chosen_kernel` | `ARG_CHOSEN_KERNEL` | the fallback that actually ran (e.g. `"generic f32 SDPA"`) |

An event carrying `fastpath_rejected_reason` (plus, ideally, `optimized_candidate`)
is a missed-fast-path signal; the event `name` is the op/node name.

**Kernel helper API** — kernel authors emit the contract in one call at the
fallback point:

```rust
use onnx_runtime_tracer::report_missed_fastpath;

// Inside a kernel, when it decides to fall back:
report_missed_fastpath(
    &ctx,                                     // the shared TraceContext
    "FusedAttention_3",                       // op / node name
    "FlashAttention",                         // optimized candidate
    "unsupported dtype fp32 (fast path is fp16/bf16 only)", // reason
    "generic f32 SDPA",                       // chosen fallback
);
```

Or attach the same keys to an existing op span:

```rust
span.set_args(Args::new().missed_fastpath("FlashAttention", reason, "generic f32 SDPA"));
```

**Reason → fix/severity mapping.** `AutoDiagnosis` derives the fix from reason
keywords: `dtype`/`precision` → cast to a supported dtype; `head_dim`/`multiple`/
`align` → pad to the required alignment; `mask` → use a supported mask form;
`EP not enabled`/`provider`/`disabled` → enable the owning EP; `threshold`/
`too small`/`below` → **`Info`** (an expected, benign fallback); anything else →
a generic-but-actionable fix. All non-threshold reasons are `Warning`.

**Follow-up seams (real emission sites to wire once the tracer is threaded into
kernels, §48.5):** `ep-cpu` `FusedAttentionKernel` (f32-only today → report for
fp16/bf16 inputs), `ep-cpu` `CpuBackend` GEMM selection (oneDNN/vendor →
`Generic` fallback), `ep-cuda` attention selection (masked/odd-shaped → no fused
cuBLAS/flash path).


### 46.7 Design Choices

| Choice | Decision | Rationale |
|--------|----------|----------|
| Format | JSONL (one event per line) | Streamable, grep-friendly, easy to parse |
| Activation by default | **Off** | Zero overhead unless opted in |
| Env var activation | **Yes** (`ORT2_TRACE_LOG`) | Debug without code change |
| Value capture | **Opt-in per node** | Full capture = huge files; selective = manageable |
| Auto-diagnosis | **Yes** | Most users can't read raw traces; give them answers |
| Trace diff/bisect tools | **Yes** | Essential for "our output differs from ORT" bugs |
| Replayable | **Yes** | Offline debugging without model/hardware access |

---

## 47. Identified Gaps & Roadmap Additions

### From vLLM:
- [ ] **LoRA serving** — per-request adapter selection, multi-LoRA batched GEMM
- [ ] **Async output processing** — detokenize pipeline doesn't block decode loop
- [ ] **Recompute-based preemption** — drop KV and recompute (faster than swap to CPU for short sequences)
- [ ] **Multi-LoRA batching** — single CUDA kernel handles different adapters in same batch
- [ ] **Frequency-based prefix cache eviction** — complement LRU with usage frequency
- [ ] **Speculative decode batch scheduling** — draft batch independent of target batch

### Quantization:
- [ ] **FP8 (E4M3/E5M2)** — H100/H200 feature, fused FP8 GEMM kernels
- [ ] **Dynamic quantization (activation)** — runtime calibration for activation quantization
- [ ] **GGUF format loading** — direct weight loading from GGUF files (llama.cpp compat)
- [ ] **GPTQ/AWQ validation** — verify group_size and packing compat with MatMulNBits

### Custom EP:
- [ ] **EP custom optimization passes** — EPs register their own fusion patterns
- [ ] **EP node claiming API** — QNN/TRT style "I want these nodes" (fallback when cost model insufficient)

### Trace & Profiling:
- [ ] **Page migration events** — full detail: page_id, content, from/to tier, duration, reason, overlap info
- [ ] **Counter tracks (Perfetto)** — continuous GPU mem, KV growth, pressure level
- [ ] **Flow events** — data dependency arrows (prefetch H2D → consuming kernel)
- [x] **CUPTI integration (§49)** — GPU kernel timing, SM occupancy, memory throughput
- [ ] **Roofline analysis per-op** — classify compute-bound vs memory-bound automatically
- [ ] **Pipeline bubble detection** — idle time in PP stages
- [ ] **Communication/compute overlap ratio** — % of transfer hidden by compute

### GenAI:
- [ ] **Component-level device placement** — vision encoder on GPU:0, LLM on GPU:1 (via existing hints, needs GenAI pipeline awareness)

---

## 48. Unified Tracing (GenAI + Runtime)

### 48.1 Problem

GenAI layer (scheduling, sampling, KV management) and runtime layer (kernel execution,
memory, transfers) each have their own events. Without unified tracing, user sees two
disconnected views and can't correlate "why was this decode step slow" with "because a
page was being fetched".

### 48.2 Crate: `onnx-runtime-tracer`

Tracing is an **independent crate** shared by both layers. Minimal dependencies (only `tracing` + protobuf).

```
onnx-runtime-tracer        ← standalone, zero runtime dependency
├── depended on by: onnx-runtime-session (runtime layer)
├── depended on by: onnx-genai-engine (genai layer)
└── depended on by: user code (custom collectors, analysis tools)
```

```toml
# workspace Cargo.toml
[workspace.dependencies]
onnx-runtime-tracer = { path = "crates/onnx-runtime-tracer" }
```

**Crate contents:**
```rust
// crates/onnx-runtime-tracer/src/lib.rs
pub struct TraceContext { .. }       // shared context (clock + collector + config)
pub trait TraceCollector { .. }      // output sink trait
pub struct TraceEvent { .. }         // unified event type
pub enum TraceFormat { .. }          // ChromeJson | PerfettoProto | Jsonl
pub enum TraceVerbosity { .. }       // Decisions | Ops | Full

pub mod perfetto;                    // Perfetto proto serialization
pub mod chrome;                      // Chrome Trace JSON serialization
pub mod jsonl;                       // JSONL format
pub mod diagnose;                    // Auto-diagnosis engine

// Built-in collectors
pub struct FileCollector { .. }      // append to file
pub struct MemoryCollector { .. }    // collect in Vec (for API)
pub struct NoopCollector;            // zero overhead (default)
```

**Why a separate crate (not inside runtime or genai):**
- Type unification: both layers use the exact same `TraceContext` type
- No circular dependency: tracer has zero knowledge of runtime or genai internals
- User-extensible: users `impl TraceCollector` without importing heavy runtime deps
- Compile speed: small crate, fast incremental builds

### 48.3 Shared Trace Context

```rust
/// Both layers share this. Single monotonic clock, single output sink.
pub struct TraceContext {
    clock: Arc<TraceClock>,
    session_id: TraceSessionId,
    collector: Arc<dyn TraceCollector>,
    format: TraceFormat,
}

pub enum TraceFormat {
    /// Chrome Trace JSON (backward compat, Perfetto reads it too).
    ChromeJson,
    /// Perfetto native protobuf (streaming, better for large traces).
    PerfettoProto,
    /// JSONL (our structured log format, grep-friendly).
    Jsonl,
}

/// Both layers write to the same collector.
pub trait TraceCollector: Send + Sync {
    fn emit(&self, event: TraceEvent);
    fn flush(&self) -> Result<()>;
}

impl TraceContext {
    /// No-op context: zero overhead when tracing disabled.
    pub fn noop() -> Self;
}
```

### 48.3 Perfetto Track Layout

When opened in Perfetto UI, the unified trace shows:

```
Process: nxrt (pid=1)
├─ Thread: genai.scheduler      → request lifecycle, batch decisions
├─ Thread: genai.sampler        → sampling, speculation, logit processing
├─ Thread: genai.kv_cache       → page alloc/evict/migrate/prefix_match
├─ Thread: runtime.compute      → kernel execution (per CUDA stream)
├─ Thread: runtime.h2d          → host-to-device transfers
├─ Thread: runtime.d2h          → device-to-host transfers
├─ Thread: runtime.optimizer    → optimization passes (load-time only)
├─ Counter: gpu_memory_mb       → continuous VRAM usage
├─ Counter: kv_pages_gpu        → pages on GPU over time
├─ Counter: batch_size          → active decode batch size
├─ Counter: tokens_per_sec      → throughput
└─ Counter: memory_pressure     → 0-1 pressure level
```

### 48.4 Flow Events (Data Dependency Arrows)

Perfetto flow events draw arrows between related events across tracks:

```rust
pub struct FlowEvent {
    flow_id: u64,
    kind: FlowKind,  // Start, Step, End
}

// Visible arrows in Perfetto:
// 1. prefetch(layer.5.weight) ──flow──→ kernel(layer.5.matmul)
// 2. kv_append(req_42) ──flow──→ attention(req_42, next step)
// 3. spec_draft ──flow──→ verify_batch ──flow──→ accept/reject
// 4. pressure_event ──flow──→ evict_lru ──flow──→ page_migrate
```

### 48.5 Integration API

```rust
impl GenAiEngine {
    pub fn new(session: InferenceSession, trace: Option<TraceContext>) -> Self {
        let trace_ctx = trace.unwrap_or(TraceContext::noop());
        // Pass same context to runtime
        session.set_trace_context(trace_ctx.clone());
        Self { session, trace: trace_ctx, .. }
    }
}

impl InferenceSession {
    /// Inject shared trace context (called by GenAI layer or user directly).
    pub fn set_trace_context(&mut self, ctx: TraceContext);
}
```

### 48.6 User API

```python
import nxrt

# Unified trace: one file, both layers
with nxrt.trace("/tmp/unified.perfetto", format="perfetto") as t:
    engine = nxrt.GenAiEngine("model.onnx", trace=t)
    for token in engine.generate_stream("Hello"):
        print(token, end="")
# Open /tmp/unified.perfetto in Perfetto UI

# Env var (no code change):
# ORT2_TRACE=/tmp/trace.perfetto ORT2_TRACE_FORMAT=perfetto python serve.py
```

```bash
# CLI
nxrt run model.onnx --trace /tmp/trace.perfetto --trace-format perfetto

# Merge with other Perfetto traces (e.g. from GitHub Copilot)
# Both use standard perfetto.protos.Trace — can view side by side in Perfetto UI
```

### 48.7 Copilot Perfetto Compatibility

If GitHub Copilot is adding Perfetto support, we ensure:
1. Standard `perfetto.protos.Trace` protobuf format
2. Track naming convention compatible for merged viewing
3. Support `trace_processor` SQL queries for custom analysis
4. Shared `flow_id` namespace for cross-tool correlation

### 48.8 Multi-Backend Architecture

One instrumentation point, multiple consumers. Code never knows which backends are active.

```
trace_event!("MatMul_0", ...)      ← single annotation in code
         │
         ▼
  ┌─────────────────────────┐
  │   TraceContext           │
  │   (single emit point)    │
  └────────────┬────────────┘
               │ fan-out via CompositeCollector
    ┌──────────┼──────────────┬────────────────┬─────────────┐
    ▼          ▼              ▼                ▼             ▼
 Perfetto   Chrome JSON    ITT API          CUPTI        Custom
 Collector  Collector      Collector        Collector    (OTel, webhook)
    │          │              │                │
    ▼          ▼              ▼                ▼
 .perfetto   trace.json    VTune/Inspector  GPU kernel   Datadog/etc.
```

#### 48.8.1 CompositeCollector

```rust
/// Fan-out: emit one event to multiple backends simultaneously.
pub struct CompositeCollector {
    collectors: Vec<Box<dyn TraceCollector>>,
}

impl CompositeCollector {
    pub fn new() -> Self {
        Self { collectors: Vec::new() }
    }

    pub fn add(&mut self, collector: Box<dyn TraceCollector>) {
        self.collectors.push(collector);
    }
}

impl TraceCollector for CompositeCollector {
    fn emit(&self, event: &TraceEvent) {
        for collector in &self.collectors {
            collector.emit(event);
        }
    }

    fn flush(&self) -> Result<()> {
        for collector in &self.collectors {
            collector.flush()?;
        }
        Ok(())
    }
}
```

#### 48.8.2 ITT Collector (Intel VTune / Inspector)

Bridges our `TraceEvent` into Intel ITT API annotations. When VTune is attached, events
become visible in its timeline. When not attached, ITT stubs out to zero overhead.

Dependency: `ittapi` crate (official Rust bindings from `intel/ittapi`, on crates.io).

```rust
use ittapi::{Domain, Task, StringHandle};

pub struct IttCollector {
    domain: Domain,
    /// Pre-registered string handles (ITT perf requirement).
    string_cache: DashMap<String, StringHandle>,
}

impl IttCollector {
    pub fn new(domain_name: &str) -> Self {
        Self {
            domain: Domain::new(domain_name),
            string_cache: DashMap::new(),
        }
    }

    fn get_or_create_handle(&self, name: &str) -> StringHandle {
        self.string_cache
            .entry(name.to_string())
            .or_insert_with(|| StringHandle::new(&self.domain, name))
            .clone()
    }
}

impl TraceCollector for IttCollector {
    fn emit(&self, event: &TraceEvent) {
        match event.phase {
            TracePhase::Begin => {
                let handle = self.get_or_create_handle(&event.name);
                Task::begin(&self.domain, handle);
            }
            TracePhase::End => {
                Task::end(&self.domain);
            }
            TracePhase::Instant => {
                let handle = self.get_or_create_handle(&event.name);
                ittapi::marker(&self.domain, handle);
            }
            _ => {} // counters — no direct ITT equivalent
        }
    }

    fn flush(&self) -> Result<()> {
        Ok(()) // ITT doesn't need explicit flush
    }
}
```

ITT also provides **JIT Profiling API** — useful for reporting NVRTC-compiled kernels
and dynamically generated code to VTune:

```rust
// When we JIT-compile an elementwise kernel via NVRTC:
ittapi::jit::notify_code_loaded(
    "fused_gelu_silu_fp16",   // symbol name VTune will show
    code_ptr,                  // pointer to compiled code
    code_size,                 // code size in bytes
);
```

#### 48.8.3 CUPTI Collector

Wraps §49 `CuptiProfiler` as a `TraceCollector`. On dispatch events, registers
correlation IDs. On flush, injects GPU kernel records back into the Perfetto trace.

```rust
pub struct CuptiCollector {
    profiler: CuptiProfiler,
}

impl TraceCollector for CuptiCollector {
    fn emit(&self, event: &TraceEvent) {
        // Register correlation on kernel dispatch
        if event.category == "compute" && event.phase == TracePhase::Begin {
            if let Some(node_id) = event.node_id {
                let corr_id = current_cuda_correlation_id();
                self.profiler.correlate(corr_id, node_id, &event.name);
            }
        }
    }

    fn flush(&self) -> Result<()> {
        // Flush CUPTI activity buffers → GPU kernel events
        // These merge into the Perfetto trace as gpu.stream* tracks
        let _gpu_records = self.profiler.stop_and_flush()?;
        Ok(())
    }
}
```

#### 48.8.4 Provider Registry (Plugin System)

```rust
/// Extensible registry of tracing backends.
pub struct TracerRegistry {
    factories: HashMap<String, Box<dyn CollectorFactory>>,
}

pub trait CollectorFactory: Send + Sync {
    /// Try to create. Returns None if backend unavailable on this system.
    /// (No CUPTI on AMD, no ITT without VTune installed, etc.)
    fn try_create(&self) -> Result<Option<Box<dyn TraceCollector>>>;
}

impl TracerRegistry {
    pub fn new() -> Self {
        let mut reg = Self { factories: HashMap::new() };
        // Built-in providers
        reg.register("perfetto", Box::new(PerfettoFactory));
        reg.register("chrome", Box::new(ChromeJsonFactory));
        reg.register("jsonl", Box::new(JsonlFactory));
        reg.register("itt", Box::new(IttFactory));       // Intel VTune
        reg.register("cupti", Box::new(CuptiFactory));   // NVIDIA GPU
        reg.register("noop", Box::new(NoopFactory));
        reg
    }

    pub fn register(&mut self, name: &str, factory: Box<dyn CollectorFactory>) {
        self.factories.insert(name.to_string(), factory);
    }

    /// Build a CompositeCollector from a list of backend names.
    pub fn build(&self, backends: &[&str]) -> Result<CompositeCollector> {
        let mut composite = CompositeCollector::new();
        for name in backends {
            let factory = self.factories.get(*name)
                .ok_or_else(|| Error::UnknownBackend(name.to_string()))?;
            if let Some(collector) = factory.try_create()? {
                composite.add(collector);
            }
            // try_create returns None → gracefully skipped
        }
        Ok(composite)
    }
}
```

#### 48.8.5 User API

```python
import nxrt

# Default: Perfetto only
session = nxrt.load("model.onnx")

# Multiple backends
session = nxrt.load("model.onnx", options={
    "trace.backends": "perfetto,itt",           # Perfetto + VTune
    # or all:
    "trace.backends": "perfetto,itt,cupti",     # all three
})

# Runtime dynamic
with nxrt.profiler.profile(backends=["perfetto", "itt", "cupti"]) as prof:
    session.run(inputs)
prof.export_perfetto("trace.perfetto")  # Perfetto file includes GPU kernel tracks
# VTune (if attached) already captured ITT events live
```

```bash
# CLI
nxrt profile model.onnx --backends perfetto,itt --output trace.perfetto

# With VTune collection wrapping us:
vtune -collect hotspots -- nxrt run model.onnx --input data.npz
# → VTune sees op-level task annotations via ITT automatically
```

#### 48.8.6 Cargo Features (Optional Dependencies)

```toml
# crates/onnx-runtime-tracer/Cargo.toml
[features]
default = ["perfetto"]
perfetto = ["dep:prost"]       # Perfetto protobuf serialization
itt = ["dep:ittapi"]           # Intel ITT API (VTune/Inspector)
cupti = []                     # CUPTI is dlopen'd, no link-time dep
chrome = []                    # Chrome Trace JSON (always available, no extra dep)
otel = ["dep:opentelemetry"]   # OpenTelemetry export (stretch)

# Python builds enable all tracing by default (batteries-included)
python-default = ["perfetto", "itt", "cupti", "chrome"]

[dependencies]
ittapi = { version = "0.4", optional = true }
prost = { version = "0.13", optional = true }
opentelemetry = { version = "0.27", optional = true }
```

#### 48.8.7 Zero-Overhead Guarantee

| Backend | No collector attached | Collector attached | Overhead source |
|---------|----------------------|-------------------|------------------|
| Perfetto | Skip (TraceContext::noop) | 3-5% | Protobuf serialization |
| ITT | 0% (stub functions) | 2-4% | VTune data collection |
| CUPTI | 0% (not dlopen'd) | 2-5% (activity) | GPU activity buffer copy |
| Chrome JSON | Skip | 5-8% | JSON formatting |

When `TraceContext::noop()` is active (no profiling), the entire trace path compiles
down to a no-op branch prediction. In release builds with LTO, the compiler eliminates
dead code paths for disabled features.

#### 48.8.8 Build-Target Defaults

Different build targets have different tracing defaults:

| Build target | Default tracing features | Rationale |
|-------------|-------------------------|----------|
| **Python (`pip install nxrt`)** | `perfetto + itt + cupti + chrome` | Batteries-included. Users expect profiling to just work. Binary size is less critical for Python packages. |
| **Rust crate (`cargo add nxrt`)** | `perfetto` only | Minimal dependencies by default. Users opt-in to `itt`/`cupti` via cargo features. |
| **C/C++ static lib** | None (noop) | Embedders control what they link. Opt-in via cmake flags: `-DNXRT_TRACE_ITT=ON`, `-DNXRT_TRACE_CUPTI=ON`. |

**Python bindings (maturin):**
```toml
# bindings/python/Cargo.toml
[dependencies]
onnx-runtime-tracer = { path = "../../crates/onnx-runtime-tracer", features = ["python-default"] }
```

**Rust library users:**
```toml
# User's Cargo.toml — opt in to what they need
[dependencies]
nxrt = { version = "0.1", features = ["itt", "cupti"] }
```

**C/C++ build (cmake):**
```bash
# Minimal (no tracing overhead)
cmake -DNXRT_TRACE=OFF ..

# With ITT (VTune support)
cmake -DNXRT_TRACE_ITT=ON ..

# Full profiling
cmake -DNXRT_TRACE_ITT=ON -DNXRT_TRACE_CUPTI=ON -DNXRT_TRACE_PERFETTO=ON ..
```

This ensures:
- **Python users** get the best out-of-box experience (profiling works immediately)
- **Rust/C++ users** get minimal binary size and dependency footprint by default
- **Everyone** can opt-in to exactly what they need without rebuilding the world

#### 48.8.9 Feature Placement: Where Each Backend Lives

Not all tracing backends belong in the same crate. Key distinction:

- **ITT = software annotation** — marks any code path (optimizer, scheduler, EP dispatch,
  loader, memory planner). Belongs in **tracer crate** itself because it instruments the
  entire runtime, not just one EP.

- **CUPTI = hardware collection** — only meaningful with NVIDIA GPU hardware. Belongs in
  **ep-cuda crate** which activates the tracer's `cupti` feature.

```
onnx-runtime-tracer
├── feature "perfetto" (default)    → Perfetto protobuf export
├── feature "itt"                   → Intel ITT API (ittapi crate)
│   └── IttCollector annotates ALL TraceEvents:
│       optimizer passes, memory planning, loader, scheduler,
│       CPU EP dispatch, GPU EP dispatch — everything.
│       (~50KB binary overhead, zero runtime cost without VTune)
├── feature "cupti"                 → CUPTI collector interface + dlopen shim
│   └── Only compiled when activated by ep-cuda
└── feature "chrome"                → Chrome Trace JSON

onnx-runtime-ep-cuda
├── depends on: tracer = { features = ["cupti"] }   ← activates CUPTI code
│   └── dlopen libcupti.so at runtime (graceful if absent)
├── owns: GPU kernel correlation, activity buffer management
└── reason: CUPTI only exists on NVIDIA hardware

onnx-runtime-ep-cpu
├── depends on: tracer (no extra features needed)
│   └── ITT already covers CPU EP dispatch via tracer-level annotation
└── does NOT need to explicitly pull ITT — tracer handles it
```

**Why ITT ≠ CUPTI placement:**

| Aspect | ITT | CUPTI |
|--------|-----|-------|
| What it does | Annotates code ("this region is MatMul") | Collects HW counters (kernel time, occupancy) |
| Scope | All code paths (CPU, GPU, optimizer, scheduler) | GPU kernels only |
| Hardware dependency | None (works everywhere) | Requires NVIDIA GPU + driver |
| Binary cost | ~50KB (ittapi static lib) | ~0 (dlopen, no link-time cost) |
| Placement | **tracer crate** | **ep-cuda crate** (activates tracer/cupti feature) |

**Python wheel implications:**

```
nxrt (base wheel, CPU-only, ~5MB):
  tracer with [perfetto, itt, chrome] ← ITT included, zero cost
  NO cupti code compiled

nxrt[cuda12] (GPU extras):
  ep-cuda activates tracer/cupti feature
  dlopen libcupti.so at runtime
```

This means:
- CPU-only users get VTune support for free (ITT in base package)
- CUPTI code never touches a non-GPU build
- GPU users get both ITT + CUPTI automatically

#### 48.8.10 Design Decisions

| Decision | Choice | Rationale |
|----------|--------|----------|
| Single emit point | **Yes** (TraceEvent) | Code annotates once, backends are config |
| Fan-out vs select-one | **Fan-out** (CompositeCollector) | Multiple tools simultaneously is normal workflow |
| ITT as optional feature | **Yes** (cargo feature `itt`) | Don't pull ittapi for non-Intel targets |
| CUPTI as dlopen | **Yes** (no link-time dep) | Works on machines without NVIDIA drivers |
| Registry extensible | **Yes** (CollectorFactory trait) | Third-party: OpenTelemetry, Datadog, custom |
| Provider unavailable | **Graceful skip** (try_create → None) | Don't crash if user requests "cupti" on AMD |
| String handle caching (ITT) | **Yes** (DashMap) | ITT perf requirement: pre-register strings |

---

## 49. CUPTI Integration (GPU Kernel-Level Tracing)

### 49.1 Problem

§48 Unified Tracing captures **runtime-level** events (which op ran, how long the dispatch took,
transfer scheduling). But it doesn't see **inside** the GPU kernel:

- Actual kernel execution time (not just dispatch→completion wall clock)
- SM occupancy (are we warp-limited or register-limited?)
- Memory throughput (are we hitting theoretical bandwidth?)
- Warp stall reasons (memory latency, execution dependency, barrier)
- Tensor Core utilization (are we actually using TC?)

Without this, the auto-tuner (§16) can detect *which* op is slow but can't diagnose *why*.

### 49.2 Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    nxrt Profiling Stack                       │
├──────────────────────────────────────────────────────────────┤
│  §48 Unified Tracing (runtime layer)                         │
│    op dispatch, transfers, scheduling, KV cache events       │
├──────────────────────────────────────────────────────────────┤
│  §49 CUPTI Layer (GPU hardware)            ← THIS SECTION    │
│    kernel timing, occupancy, memory BW, warp stalls          │
├──────────────────────────────────────────────────────────────┤
│  CUPTI Activity API + Callback API (NVIDIA driver)           │
└──────────────────────────────────────────────────────────────┘
```

Two CUPTI APIs we use:

1. **Activity API** — async buffer of completed GPU activities (kernel, memcpy, memset)
   Low overhead (~2-5%), always-on capable for production profiling.

2. **Callback API** — synchronous hooks on kernel launch (for correlating dispatch → kernel)
   Higher overhead, used only in detailed profiling mode.

### 49.3 Rust Integration

```rust
// crates/onnx-runtime-tracer/src/cupti.rs

use std::ffi::c_void;

/// CUPTI profiling session. Wraps the CUPTI Activity API.
pub struct CuptiProfiler {
    /// Whether CUPTI is available (dlopen'd at runtime, not linked).
    available: bool,
    /// Activity buffer pool.
    buffer_pool: BufferPool,
    /// Correlation map: correlationId → (NodeId, op_type)
    correlation_map: DashMap<u32, KernelCorrelation>,
    /// Collected kernel records.
    records: Mutex<Vec<GpuKernelRecord>>,
}

/// One GPU kernel execution record.
#[derive(Debug, Clone)]
pub struct GpuKernelRecord {
    /// Correlation back to runtime op dispatch.
    pub node_id: Option<NodeId>,
    pub op_type: String,
    /// Kernel identity.
    pub kernel_name: String,
    /// Timing (GPU clock, nanoseconds).
    pub start_ns: u64,
    pub end_ns: u64,
    pub duration_ns: u64,
    /// Launch config.
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub shared_memory_bytes: u32,
    pub registers_per_thread: u32,
    /// Stream.
    pub stream_id: u32,
    /// Occupancy (computed from launch config + device props).
    pub theoretical_occupancy: f32,   // 0.0 - 1.0
    pub achieved_occupancy: Option<f32>,  // requires metric collection
}

/// Hardware performance metrics (from CUPTI Profiling API / PM sampling).
#[derive(Debug, Clone)]
pub struct GpuKernelMetrics {
    pub kernel_name: String,
    pub node_id: Option<NodeId>,
    /// Compute
    pub sm_efficiency: f32,           // % of SMs active
    pub achieved_occupancy: f32,      // % of max warps
    pub tensor_core_utilization: f32, // % of TC active cycles
    pub flop_count_sp: u64,           // single-precision FLOPs
    pub flop_count_hp: u64,           // half-precision FLOPs
    /// Memory
    pub dram_read_bytes: u64,
    pub dram_write_bytes: u64,
    pub dram_throughput_pct: f32,     // % of theoretical peak
    pub l2_hit_rate: f32,
    /// Stall reasons
    pub stall_memory_dependency: f32, // % cycles stalled on memory
    pub stall_execution_dependency: f32,
    pub stall_barrier: f32,
    pub stall_not_selected: f32,
}

impl CuptiProfiler {
    /// Initialize. Attempts dlopen of libcupti.so at runtime.
    /// Returns Ok with available=false if CUPTI not present (graceful degradation).
    pub fn new() -> Result<Self> {
        let available = unsafe { try_load_cupti() };
        Ok(Self {
            available,
            buffer_pool: BufferPool::new(8, 1 << 20), // 8 buffers × 1MB
            correlation_map: DashMap::new(),
            records: Mutex::new(Vec::new()),
        })
    }

    /// Start activity tracing (low overhead mode).
    pub fn start_activity_tracing(&mut self) -> Result<()> {
        if !self.available { return Ok(()); }
        // Enable: CUPTI_ACTIVITY_KIND_KERNEL, CUPTI_ACTIVITY_KIND_MEMCPY,
        //         CUPTI_ACTIVITY_KIND_MEMSET, CUPTI_ACTIVITY_KIND_OVERHEAD
        unsafe {
            cupti_activity_enable(CUPTI_ACTIVITY_KIND_KERNEL)?;
            cupti_activity_enable(CUPTI_ACTIVITY_KIND_MEMCPY)?;
            cupti_activity_register_callbacks(
                Self::buffer_requested,
                Self::buffer_completed,
            )?;
        }
        Ok(())
    }

    /// Stop tracing and flush remaining buffers.
    pub fn stop_and_flush(&mut self) -> Result<Vec<GpuKernelRecord>> {
        if !self.available { return Ok(vec![]); }
        unsafe { cupti_activity_flush_all()?; }
        let records = std::mem::take(self.records.lock().unwrap().as_mut());
        Ok(records)
    }

    /// Register a kernel launch correlation (called from EP dispatch).
    pub fn correlate(&self, correlation_id: u32, node_id: NodeId, op_type: &str) {
        self.correlation_map.insert(correlation_id, KernelCorrelation {
            node_id,
            op_type: op_type.to_string(),
        });
    }

    /// Collect detailed metrics for specific kernels (higher overhead).
    /// Uses CUPTI Profiling API (range-based, replay mode).
    pub fn collect_metrics(
        &self,
        kernel_names: &[&str],
        metrics: &[CuptiMetric],
        num_runs: usize,
    ) -> Result<Vec<GpuKernelMetrics>> {
        // CUPTI Profiling API requires kernel replay for counter collection
        // 1. Create profiler config with requested metrics
        // 2. Begin pass (may need multiple passes for many metrics)
        // 3. Run kernels (caller invokes model execution)
        // 4. End pass, decode counters
        todo!()
    }
}

/// Metrics we can request from CUPTI Profiling API.
pub enum CuptiMetric {
    SmEfficiency,
    AchievedOccupancy,
    TensorCoreUtilization,
    DramThroughput,
    L2HitRate,
    WarpStallReasons,
    FlopCount,
}
```

### 49.4 Correlation: Runtime Op → GPU Kernel

The key challenge is linking runtime-level op dispatch to CUPTI's kernel records:

```rust
// In EP dispatch (onnx-runtime-ep-cuda)
impl CudaExecutionProvider {
    fn execute_kernel(&self, node: &Node, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        // Get CUPTI correlation ID from current CUDA context
        let correlation_id = cuda_get_next_correlation_id();

        // Register correlation before launch
        if let Some(profiler) = self.cupti_profiler.as_ref() {
            profiler.correlate(correlation_id, node.id, &node.op_type);
        }

        // Launch kernel (CUPTI automatically tags it with correlation_id)
        self.kernels[&node.op_type].launch(inputs, outputs, self.stream)?;

        Ok(())
    }
}
```

In Perfetto output, this gives us nested spans:
```
[runtime.compute]  ──── MatMul_0 (dispatch, 55μs) ────
[runtime.gpu]          ── volta_sgemm_128x64 (kernel, 42μs) ──
                              ↑ correlated via correlation_id
```

### 49.5 Roofline Analysis (Auto-Classification)

```rust
/// Automatically classify each kernel as compute-bound or memory-bound.
pub struct RooflineAnalyzer {
    /// Device peak performance.
    peak_flops_fp16: f64,      // e.g. 312 TFLOPS for H100 TC
    peak_flops_fp32: f64,      // e.g. 67 TFLOPS for H100
    peak_bandwidth: f64,        // e.g. 3.35 TB/s for H100 HBM3e
}

#[derive(Debug, Clone)]
pub struct RooflineResult {
    pub kernel_name: String,
    pub node_id: NodeId,
    /// Arithmetic intensity: FLOPs / bytes accessed
    pub arithmetic_intensity: f64,
    /// Achieved performance
    pub achieved_tflops: f64,
    /// Classification
    pub bound: BoundType,
    /// How far from roofline (1.0 = at roofline, 0.5 = half of peak)
    pub efficiency: f64,
    /// Actionable suggestion
    pub suggestion: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BoundType {
    ComputeBound,     // AI > ridge point → limited by FLOPS
    MemoryBound,      // AI < ridge point → limited by bandwidth
    LaunchBound,      // kernel too short, dominated by launch overhead
    Balanced,         // near ridge point
}

impl RooflineAnalyzer {
    pub fn analyze(&self, metrics: &GpuKernelMetrics) -> RooflineResult {
        let bytes_accessed = metrics.dram_read_bytes + metrics.dram_write_bytes;
        let flops = metrics.flop_count_hp + metrics.flop_count_sp;
        let ai = flops as f64 / bytes_accessed.max(1) as f64;

        // Ridge point: where compute ceiling meets memory ceiling
        let ridge_point = self.peak_flops_fp16 / self.peak_bandwidth;

        let bound = if ai > ridge_point * 1.2 {
            BoundType::ComputeBound
        } else if ai < ridge_point * 0.8 {
            BoundType::MemoryBound
        } else {
            BoundType::Balanced
        };

        let theoretical_peak = if ai > ridge_point {
            self.peak_flops_fp16
        } else {
            ai * self.peak_bandwidth
        };
        let achieved = flops as f64 / metrics.duration_ns as f64 * 1e9;
        let efficiency = achieved / theoretical_peak;

        let suggestion = match bound {
            BoundType::MemoryBound if efficiency < 0.5 => {
                Some("Memory-bound with low bandwidth utilization. Consider: \
                      (1) fuse with adjacent ops to reduce memory traffic, \
                      (2) use in-place operations, (3) quantize to reduce data size".into())
            }
            BoundType::ComputeBound if efficiency < 0.5 => {
                Some("Compute-bound with low SM utilization. Consider: \
                      (1) increase occupancy (reduce registers/shared mem), \
                      (2) use Tensor Cores (fp16/bf16/int8), \
                      (3) increase tile size for better data reuse".into())
            }
            BoundType::LaunchBound => {
                Some("Kernel too short — launch overhead dominates. \
                      Consider: (1) fuse with neighbors, (2) CUDAGraph capture".into())
            }
            _ => None,
        };

        RooflineResult { kernel_name: metrics.kernel_name.clone(), node_id: metrics.node_id.unwrap(), arithmetic_intensity: ai, achieved_tflops: achieved / 1e12, bound, efficiency, suggestion }
    }
}
```

### 49.6 Profiling Modes

| Mode | Overhead | What you get | Use case |
|------|----------|-------------|----------|
| **Off** | 0% | Nothing | Production |
| **Activity** | 2-5% | Kernel timing + memcpy + correlation | Default profiling (`nxrt profile`) |
| **Detailed** | 10-30% | Activity + occupancy + basic metrics | Bottleneck diagnosis |
| **Metrics** | 50-200% (replay) | Full PM counters (roofline, stalls, TC util) | Deep optimization |

```python
import nxrt

session = nxrt.load("model.onnx", device="cuda")

# Activity mode (low overhead)
with nxrt.profiler.profile(gpu="activity") as prof:
    session.run(inputs)
prof.export_perfetto("trace.perfetto")

# Detailed mode
with nxrt.profiler.profile(gpu="detailed") as prof:
    session.run(inputs)
for kernel in prof.gpu_kernels():
    print(f"{kernel.op_type}: {kernel.duration_ns/1000:.1f}μs, occupancy={kernel.theoretical_occupancy:.0%}")

# Metrics mode (roofline)
with nxrt.profiler.profile(gpu="metrics", metrics=["roofline"]) as prof:
    session.run(inputs)  # may run multiple times internally (CUPTI replay)
for r in prof.roofline():
    print(f"{r.kernel_name}: AI={r.arithmetic_intensity:.1f}, {r.bound}, efficiency={r.efficiency:.0%}")
    if r.suggestion:
        print(f"  → {r.suggestion}")
```

### 49.7 CLI Integration

```bash
# Quick profile (activity mode)
nxrt profile model.onnx --inputs data.npz --gpu activity --output trace.perfetto

# Roofline analysis
nxrt profile model.onnx --inputs data.npz --gpu metrics --roofline
# Output:
#   MatMul_0    : AI=128.5  compute-bound  efficiency=72%
#   LayerNorm_3 : AI=2.1    memory-bound   efficiency=45% → fuse with residual add
#   Softmax_1   : AI=3.8    memory-bound   efficiency=61%
#   Add_7       : AI=0.25   launch-bound   → CUDAGraph or fuse

# Top-N slowest GPU kernels
nxrt profile model.onnx --inputs data.npz --gpu detailed --top 10
```

### 49.8 Perfetto Integration

CUPTI data merges into the same Perfetto trace as §48:

```
Process: nxrt (pid=1)
├─ Thread: genai.scheduler        → request lifecycle
├─ Thread: runtime.compute         → op dispatch (§48)
├─ Thread: runtime.gpu.stream0     → GPU kernel execution (§49, CUPTI)  ← NEW
├─ Thread: runtime.gpu.stream1     → GPU memcpy (§49, CUPTI)            ← NEW
├─ Counter: gpu.sm_occupancy       → achieved occupancy over time        ← NEW
├─ Counter: gpu.memory_bw_util     → % of peak HBM bandwidth            ← NEW
├─ Counter: gpu.tensor_core_util   → TC utilization                      ← NEW
└─ Flow: dispatch → kernel         → correlation arrows                  ← NEW
```

### 49.9 Dynamic Loading (No Hard Dependency)

```rust
/// CUPTI is dlopen'd at runtime. Runtime works without it.
pub struct CuptiLibrary {
    lib: Option<libloading::Library>,
    // Function pointers (populated on successful dlopen)
    activity_enable: Option<unsafe extern "C" fn(kind: u32) -> u32>,
    activity_flush: Option<unsafe extern "C" fn(flag: u32) -> u32>,
    // ...
}

impl CuptiLibrary {
    pub fn try_load() -> Self {
        let lib = unsafe { libloading::Library::new("libcupti.so") }
            .or_else(|_| unsafe { libloading::Library::new("libcupti.dylib") })
            .ok();
        // Resolve symbols if loaded...
        Self { lib, /* ... */ }
    }

    pub fn is_available(&self) -> bool { self.lib.is_some() }
}
```

### 49.10 Design Decisions

| Decision | Choice | Rationale |
|----------|--------|----------|
| Hard link vs dlopen | **dlopen** | Runtime works on machines without CUDA/CUPTI |
| Activity API vs Callback | **Activity primary** + callback for correlation | Activity is lower overhead |
| Metrics collection | **Replay-based** (CUPTI Profiling API) | Counters require kernel replay; explicit opt-in |
| Roofline auto-analysis | **Yes** | Key output for auto-tuner (§16) |
| Integration with §48 | **Same Perfetto trace** | One file, unified view |
| Correlation mechanism | **CUPTI correlation_id** mapped to NodeId | Standard CUPTI approach |
| Overhead guarantee | **Activity < 5%** in default profiling | Usable in near-production |

---

## 50. CPU Hardware Detection (cpuinfo)

### 50.1 Problem

The CPU EP needs to select optimal kernel implementations at runtime:
- AVX-512 VNNI matmul vs AVX2+FMA vs NEON
- Tile sizes tuned to L2/L3 cache capacity
- Thread pool affinity respecting core topology (P-cores vs E-cores, NUMA)
- AMX (Sapphire Rapids+) detection for int8/bf16 matmul

Without hardware detection, we either ship the lowest-common-denominator kernel
or require users to set flags manually.

### 50.2 Detection Layer

```rust
// crates/onnx-runtime-cpuinfo/src/lib.rs

/// CPU hardware capabilities, detected once at initialization.
/// Thread-safe, immutable after init.
#[derive(Debug, Clone)]
pub struct CpuInfo {
    // === ISA Features ===
    pub isa: IsaFeatures,
    // === Cache Hierarchy ===
    pub caches: Vec<CacheLevel>,
    // === Core Topology ===
    pub topology: CpuTopology,
    // === Vendor / Microarchitecture ===
    pub vendor: CpuVendor,
    pub microarch: Microarchitecture,
    pub model_name: String,
}

/// Instruction set features.
#[derive(Debug, Clone, Default)]
pub struct IsaFeatures {
    // x86
    pub avx: bool,
    pub avx2: bool,
    pub fma: bool,
    pub avx512f: bool,
    pub avx512bw: bool,
    pub avx512vnni: bool,      // int8 dot product
    pub avx512bf16: bool,      // bf16 support
    pub amx_int8: bool,        // Sapphire Rapids+ tile matmul
    pub amx_bf16: bool,
    pub amx_fp16: bool,        // Granite Rapids+
    // ARM
    pub neon: bool,
    pub sve: bool,
    pub sve2: bool,
    pub sve_bitwidth: Option<u32>,  // 128/256/512
    pub dotprod: bool,          // SDOT/UDOT (int8)
    pub fp16_arith: bool,       // FEAT_FP16 (native f16 compute)
    pub i8mm: bool,             // int8 matrix multiply
    pub bf16: bool,             // FEAT_BF16
    pub sme: bool,              // Scalable Matrix Extension (Apple M4+, Arm v9.2)
}

/// Cache level information.
#[derive(Debug, Clone)]
pub struct CacheLevel {
    pub level: u8,              // 1, 2, 3
    pub kind: CacheKind,        // Instruction, Data, Unified
    pub size_bytes: usize,      // total size
    pub line_size: usize,       // typically 64 bytes
    pub associativity: u8,
    pub shared_by_cores: usize, // how many cores share this cache
}

/// Core topology.
#[derive(Debug, Clone)]
pub struct CpuTopology {
    pub physical_cores: usize,
    pub logical_cores: usize,
    pub packages: usize,            // sockets
    pub cores: Vec<CoreInfo>,
    pub numa_nodes: Vec<NumaNode>,
}

#[derive(Debug, Clone)]
pub struct CoreInfo {
    pub core_id: usize,
    pub package_id: usize,
    pub core_type: CoreType,        // Performance / Efficiency / Unknown
    pub logical_processors: Vec<usize>,  // hyperthreads on this core
    pub l2_cache_id: usize,         // which L2 this core uses
    pub l3_cache_id: Option<usize>, // which L3 (NUMA-aware)
}

#[derive(Debug, Clone, PartialEq)]
pub enum CoreType {
    Performance,  // P-core (Intel hybrid), Firestorm (Apple)
    Efficiency,   // E-core (Intel hybrid), Icestorm (Apple)
    Unknown,      // homogeneous or undetectable
}

#[derive(Debug, Clone)]
pub struct NumaNode {
    pub node_id: usize,
    pub core_ids: Vec<usize>,
    pub memory_size_bytes: Option<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CpuVendor {
    Intel, Amd, Apple, Arm, Qualcomm, Unknown(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Microarchitecture {
    // Intel
    SapphireRapids, GraniteRapids, RaptorLake, AlderLake,
    // AMD
    Zen4, Zen5,
    // Apple
    M1, M2, M3, M4,
    // ARM
    CortexA78, CortexX4, Neoverse_V2,
    // Other
    Unknown,
}
```

### 50.3 Detection Implementation

```rust
impl CpuInfo {
    /// Detect all CPU info. Called once at runtime init.
    /// Platform-specific internals:
    ///   - x86: CPUID instruction (leaf 0, 1, 4, 7, 11, ...)
    ///   - ARM Linux: /proc/cpuinfo + HWCAP/HWCAP2 from getauxval()
    ///   - ARM macOS: sysctlbyname("hw.optional.*")
    ///   - Cache: CPUID leaf 4 (x86), sysfs /sys/devices/system/cpu/cpu0/cache/ (Linux)
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        { Self::detect_x86() }
        #[cfg(target_arch = "aarch64")]
        { Self::detect_arm() }
    }

    #[cfg(target_arch = "x86_64")]
    fn detect_x86() -> Self {
        use std::arch::x86_64::__cpuid;

        // Leaf 0: vendor string
        let leaf0 = unsafe { __cpuid(0) };
        let vendor = parse_vendor(leaf0.ebx, leaf0.ecx, leaf0.edx);

        // Leaf 1: family/model/stepping
        let leaf1 = unsafe { __cpuid(1) };
        let microarch = identify_microarch(vendor, leaf1.eax);

        // Leaf 7: extended features (AVX2, AVX512, AMX)
        let leaf7 = unsafe { __cpuid(7) };
        let isa = IsaFeatures {
            avx2: leaf7.ebx & (1 << 5) != 0,
            avx512f: leaf7.ebx & (1 << 16) != 0,
            avx512vnni: leaf7.ecx & (1 << 11) != 0,
            amx_int8: leaf7.edx & (1 << 25) != 0,
            amx_bf16: leaf7.edx & (1 << 22) != 0,
            // ...
            ..Default::default()
        };

        // Leaf 4: cache topology
        let caches = detect_x86_caches();

        // Leaf 0xB/0x1F: core topology
        let topology = detect_x86_topology();

        Self { isa, caches, topology, vendor, microarch, model_name: read_brand_string() }
    }

    #[cfg(target_arch = "aarch64")]
    fn detect_arm() -> Self {
        #[cfg(target_os = "linux")]
        {
            // getauxval(AT_HWCAP) / AT_HWCAP2 for ISA features
            // /sys/devices/system/cpu/ for topology
            // /proc/cpuinfo for model name
            todo!()
        }
        #[cfg(target_os = "macos")]
        {
            // sysctlbyname for everything
            // hw.optional.arm.FEAT_FP16, hw.optional.arm.FEAT_BF16, etc.
            // hw.perflevel0.physicalcpu (P-cores), hw.perflevel1.physicalcpu (E-cores)
            todo!()
        }
    }
}
```

### 50.4 Usage in CPU EP

```rust
// crates/onnx-runtime-ep-cpu/src/provider.rs

impl CpuExecutionProvider {
    pub fn new() -> Self {
        let cpu = CpuInfo::detect();

        // Select kernel variants based on detected ISA
        let matmul_kernel = if cpu.isa.amx_int8 {
            MatMulKernel::Amx
        } else if cpu.isa.avx512vnni {
            MatMulKernel::Avx512Vnni
        } else if cpu.isa.avx2 && cpu.isa.fma {
            MatMulKernel::Avx2Fma
        } else if cpu.isa.neon && cpu.isa.dotprod {
            MatMulKernel::NeonDotprod
        } else {
            MatMulKernel::Generic
        };

        // Configure tiling based on cache hierarchy
        let l2_size = cpu.caches.iter()
            .find(|c| c.level == 2 && c.kind != CacheKind::Instruction)
            .map(|c| c.size_bytes)
            .unwrap_or(256 * 1024);

        let tile_config = TileConfig::optimal_for_cache(l2_size);

        // Configure thread pool based on topology
        let thread_pool = ThreadPoolConfig {
            num_threads: cpu.topology.physical_cores,  // no HT by default
            pin_to_cores: true,
            prefer_p_cores: true,  // Intel hybrid: use P-cores for compute
            numa_aware: cpu.topology.numa_nodes.len() > 1,
        };

        Self { cpu, matmul_kernel, tile_config, thread_pool, /* ... */ }
    }
}
```

### 50.5 Usage in Cost Model (§6)

```rust
// Inform the cost model with detected hardware capabilities
impl CostModel {
    pub fn from_detected_hardware(cpu: &CpuInfo, gpu: Option<&GpuInfo>) -> Self {
        let cpu_profile = DeviceProfile {
            // Estimate peak throughput from microarchitecture
            compute_throughput: estimate_cpu_peak_flops(cpu),
            // Estimate memory bandwidth from DDR generation + channels
            memory_bandwidth: estimate_cpu_bandwidth(cpu),
            // Cache-aware: operations fitting in L2 are faster
            cache_hierarchy: cpu.caches.clone(),
        };
        // ...
    }
}

fn estimate_cpu_peak_flops(cpu: &CpuInfo) -> HashMap<DataType, f64> {
    let cores = cpu.topology.physical_cores as f64;
    let freq_ghz = 3.5; // TODO: read from sysfs or estimate from microarch

    let mut throughput = HashMap::new();

    if cpu.isa.avx512f {
        // AVX-512: 32 fp32 FMA/cycle/core (2 FMA units × 16 elements)
        throughput.insert(DataType::Float32, cores * freq_ghz * 32.0 * 1e9);
    } else if cpu.isa.avx2 && cpu.isa.fma {
        // AVX2+FMA: 16 fp32 FMA/cycle/core (2 FMA units × 8 elements)
        throughput.insert(DataType::Float32, cores * freq_ghz * 16.0 * 1e9);
    }

    if cpu.isa.amx_int8 {
        // AMX: 2048 int8 ops/cycle (16×64 tile, 1 tile/cycle)
        throughput.insert(DataType::Int8, cores * freq_ghz * 2048.0 * 1e9);
    }

    throughput
}
```

### 50.6 Tiling Strategy (Cache-Aware)

```rust
pub struct TileConfig {
    pub tile_m: usize,
    pub tile_n: usize,
    pub tile_k: usize,
}

impl TileConfig {
    /// Choose tile sizes so that working set fits in L2.
    /// Goal: A_tile + B_tile + C_tile ≤ L2_size × occupancy_target
    pub fn optimal_for_cache(l2_bytes: usize) -> Self {
        // Target: use ~75% of L2 for tiles (leave room for other data)
        let budget = l2_bytes * 3 / 4;

        // For fp32 GEMM: A[M,K] + B[K,N] + C[M,N] in bytes
        // Heuristic: square-ish tiles, K=256 is good for most architectures
        let tile_k = 256;
        // Remaining budget split between M and N dimensions
        // (M*K + K*N + M*N) * 4 bytes ≤ budget
        // For simplicity: M=N, solve (2*M*K + M*M)*4 ≤ budget
        let m = ((budget / 4 - tile_k * tile_k) as f64 / (2.0 * tile_k as f64)).sqrt() as usize;
        let tile_m = m.clamp(32, 512).next_power_of_two();
        let tile_n = tile_m;

        Self { tile_m, tile_n, tile_k }
    }
}
```

### 50.7 Thread Pool Topology Awareness

```rust
pub struct TopologyAwareThreadPool {
    /// Worker threads, pinned to cores.
    workers: Vec<WorkerThread>,
    /// Scheduling: NUMA-local work first.
    scheduler: WorkScheduler,
}

impl TopologyAwareThreadPool {
    pub fn from_cpu_info(cpu: &CpuInfo, config: &ThreadPoolConfig) -> Self {
        let mut workers = Vec::new();

        // Sort cores: P-cores first (Intel hybrid)
        let mut cores: Vec<_> = cpu.topology.cores.iter().collect();
        if config.prefer_p_cores {
            cores.sort_by_key(|c| match c.core_type {
                CoreType::Performance => 0,
                CoreType::Unknown => 1,
                CoreType::Efficiency => 2,
            });
        }

        // Spawn workers on selected cores
        for core in cores.iter().take(config.num_threads) {
            let worker = WorkerThread::spawn_pinned(core.logical_processors[0]);
            workers.push(worker);
        }

        Self {
            workers,
            scheduler: WorkScheduler::new(cpu.topology.numa_nodes.clone()),
        }
    }

    /// Parallel MatMul: partition M dimension across workers.
    /// NUMA-aware: each worker operates on memory local to its NUMA node.
    pub fn parallel_matmul(
        &self,
        a: &TensorView,  // [M, K]
        b: &TensorView,  // [K, N]
        c: &mut TensorMut,  // [M, N]
        tile_config: &TileConfig,
    ) {
        let m = a.shape()[0];
        let chunk_size = m.div_ceil(self.workers.len());

        self.scheduler.parallel_for(0..m, chunk_size, |worker_id, range| {
            // Each worker processes its chunk of M rows
            tiled_matmul_chunk(a, b, c, range, tile_config);
        });
    }
}
```

### 50.8 Library Choice

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| **pytorch/cpuinfo (C)** | Battle-tested, covers edge cases, mock tests | C FFI, extra dependency, may be stale | Reference, not direct dep |
| **raw-cpuid (Rust)** | Pure Rust, x86 only, well maintained | No ARM, no topology | Use for x86 CPUID |
| **std::arch::is_*_detected!** | Stdlib, zero dep | Features only, no cache/topology | Supplement |
| **sysfs/procfs parsing** | Full topology on Linux | Linux-only, parsing brittle | Linux topology backend |
| **sysctlbyname** | Full info on macOS | macOS-only | macOS backend |
| **Our own crate** | Exactly what we need, Rust-native | Dev effort | ✅ Decision |

**Decision:** Write our own `onnx-runtime-cpuinfo` crate.
- Use `raw-cpuid` for x86 CPUID leaves (don't reinvent leaf parsing)
- Use `std::arch` macros as cross-check
- Linux topology from sysfs, macOS from sysctl
- Inspired by pytorch/cpuinfo's structure but Rust-native, no C FFI
- Focus on what we need: ISA, cache, topology. Skip SoC identification for phones.

### 50.9 Crate Structure

```
crates/onnx-runtime-cpuinfo/
├── Cargo.toml
└── src/
    ├── lib.rs              # CpuInfo + public types
    ├── isa.rs              # IsaFeatures detection
    ├── cache.rs            # Cache hierarchy detection
    ├── topology.rs         # Core topology, NUMA
    ├── vendor.rs           # Vendor + microarch identification
    ├── x86/
    │   ├── mod.rs
    │   ├── cpuid.rs        # Raw CPUID leaf access (uses raw-cpuid crate)
    │   └── topology.rs     # x86-specific topology (leaf 0xB, 0x1F)
    ├── arm/
    │   ├── mod.rs
    │   ├── linux.rs        # /proc/cpuinfo + getauxval + sysfs
    │   └── macos.rs        # sysctlbyname
    └── tests/
        ├── mock_sapphire_rapids.rs
        ├── mock_apple_m4.rs
        └── mock_neoverse_v2.rs
```

### 50.10 Design Decisions

| Decision | Choice | Rationale |
|----------|--------|----------|
| Own crate vs cpuinfo FFI | **Own crate** | Pure Rust, no C build complexity, exactly our needs |
| Detection timing | **Once at init** (lazy_static/OnceLock) | Immutable after detection, zero runtime overhead |
| Fallback on unknown | **Conservative defaults** | Unknown CPU → generic kernels, default tile sizes. Never crash. |
| P/E core detection | **Yes** | Intel 12th+ gen and Apple Silicon need this for thread scheduling |
| NUMA awareness | **Yes** | Multi-socket servers are deployment targets |
| Mock testing | **Yes** (inject fake CPUID) | Can't run CI on every microarchitecture |

---

## 51. LoRA Serving

### 51.1 Problem

Serving multiple fine-tuned adapters from one base model. Each user/request may need
a different LoRA. Loading a full model per adapter = wasteful.

### 49.2 Architecture

```rust
pub struct LoraManager {
    /// Base model weights (shared across all requests).
    base_weights: Arc<WeightStore>,
    /// Loaded LoRA adapters (hot cache).
    adapters: HashMap<AdapterId, LoraAdapter>,
    /// Max adapters in GPU memory simultaneously.
    max_active: usize,
    /// LRU eviction for cold adapters.
    lru: LruCache<AdapterId>,
}

pub struct LoraAdapter {
    id: AdapterId,
    /// LoRA A matrices (low-rank, per target module).
    a_weights: HashMap<String, Tensor>,  // module_name → [rank, in_features]
    /// LoRA B matrices.
    b_weights: HashMap<String, Tensor>,  // module_name → [out_features, rank]
    /// Scaling factor.
    alpha: f32,
    rank: usize,
    /// Which modules this adapter targets.
    target_modules: Vec<String>,
}
```

### 49.3 Three Approaches

**A. On-the-fly merge (simple, per-request overhead):**
```rust
/// Before inference: W_effective = W_base + (alpha/rank) * B @ A
/// Modify weight in-place (or use separate buffer).
impl LoraManager {
    fn apply_lora(&self, base: &Tensor, adapter: &LoraAdapter, module: &str) -> Tensor {
        let a = &adapter.a_weights[module];
        let b = &adapter.b_weights[module];
        let scale = adapter.alpha / adapter.rank as f32;
        // base + scale * (B @ A)  — one extra matmul per adapted layer
        base + scale * b.matmul(a)
    }
}
```
Pros: Simple, correct. Cons: Extra GEMM per layer per request.

**B. Batched LoRA GEMM (vLLM's approach — multi-adapter in one kernel):**
```rust
/// Multiple requests in same batch, each with different adapter.
/// Single CUDA kernel handles all adapters using indirect addressing.
pub struct BatchedLoraKernel {
    /// Per-request adapter index: batch[i] uses adapter_indices[i].
    adapter_indices: Vec<u32>,
    /// Stacked A matrices: [num_adapters, rank, in_features]
    stacked_a: Tensor,
    /// Stacked B matrices: [num_adapters, out_features, rank]
    stacked_b: Tensor,
}

impl Kernel for BatchedLoraKernel {
    fn execute(&self, ctx: &mut KernelContext) -> Result<()> {
        // CUDA kernel:
        // For each batch element i:
        //   adapter = adapter_indices[i]
        //   output[i] = base_output[i] + scale * stacked_b[adapter] @ (stacked_a[adapter] @ input[i])
        // All in one kernel launch (no per-request overhead).
    }
}
```
Pros: Batch-efficient, one kernel for entire batch. Cons: Custom CUDA kernel needed.

**C. Weight decomposition caching (pre-merge, amortized):**
```rust
/// For frequently-used adapters: pre-compute merged weight, cache it.
impl LoraManager {
    fn get_merged_weight(&mut self, module: &str, adapter: AdapterId) -> &Tensor {
        self.merge_cache.get_or_insert((module, adapter), || {
            let base = &self.base_weights[module];
            let a = &self.adapters[adapter].a_weights[module];
            let b = &self.adapters[adapter].b_weights[module];
            base + (alpha / rank) * b.matmul(a)
        })
    }
}
```
Pros: Zero per-request overhead for hot adapters. Cons: Memory (one full weight per adapter per module).

### 49.4 Our Strategy: Hybrid

```rust
pub enum LoraStrategy {
    /// Few adapters, high throughput: pre-merge weights.
    PreMerge,
    /// Many adapters, mixed batch: batched LoRA kernel.
    BatchedKernel,
    /// Fallback: on-the-fly merge.
    OnTheFly,
    /// Auto: choose based on adapter count and request patterns.
    Auto,
}
```

Auto logic:
- ≤4 active adapters → PreMerge (fast, fits in memory)
- 5-32 adapters, mixed batch → BatchedKernel
- \>32 or cold adapters → OnTheFly for cold, BatchedKernel for hot

### 49.5 User API

```python
engine = nxrt.GenAiEngine("base_model.onnx")

# Load adapters
engine.load_lora("coding_assistant", "lora_coding.safetensors")
engine.load_lora("creative_writer", "lora_creative.safetensors")

# Per-request adapter selection
output = engine.generate("Write a poem", lora="creative_writer")
output = engine.generate("def quicksort", lora="coding_assistant")

# Mixed batch (different adapters in same batch)
batch = [
    {"prompt": "Write a poem", "lora": "creative_writer"},
    {"prompt": "def quicksort", "lora": "coding_assistant"},
    {"prompt": "Hello", "lora": None},  # base model, no adapter
]
outputs = engine.generate_batch(batch)
```

---

## 52. Async Output Processing

### 50.1 Problem

After sampling a token, several steps happen before the user sees it:
1. Detokenization (token_id → text)
2. Stop sequence checking
3. Streaming response write
4. Metrics update

If these happen synchronously, they block the next decode step.
vLLM found that detokenization alone can add 0.5-1ms per step.

### 50.2 Design: Async Output Pipeline

```rust
pub struct AsyncOutputPipeline {
    /// Channel: decode loop sends tokens, pipeline processes async.
    sender: mpsc::Sender<OutputEvent>,
    /// Background thread handles detokenization + streaming.
    worker: JoinHandle<()>,
}

pub enum OutputEvent {
    /// New token sampled. Needs detokenization.
    Token { request_id: RequestId, token_id: TokenId, logprob: Option<f32> },
    /// Sequence finished.
    Done { request_id: RequestId, finish_reason: FinishReason },
    /// Sequence aborted (cancelled, error).
    Abort { request_id: RequestId, reason: String },
}

impl AsyncOutputPipeline {
    fn worker_loop(receiver: mpsc::Receiver<OutputEvent>, tokenizer: Arc<Tokenizer>) {
        for event in receiver {
            match event {
                OutputEvent::Token { request_id, token_id, logprob } => {
                    // Detokenize (can be expensive for multi-byte UTF-8)
                    let text = tokenizer.decode_incremental(request_id, token_id);
                    // Check stop sequences
                    let stopped = check_stop_sequences(&text, &request.stop);
                    // Stream to client (non-blocking write to response channel)
                    request.stream_sender.try_send(StreamChunk { text, logprob, stopped });
                    // Update metrics
                    metrics.tokens_generated.inc();
                }
                _ => { /* handle done/abort */ }
            }
        }
    }
}
```

**Decode loop (unblocked):**
```rust
impl GenAiEngine {
    fn decode_step(&mut self) -> Result<()> {
        // 1. Run model forward pass (GPU)
        let logits = self.session.run_decode_step(&inputs)?;

        // 2. Sample tokens (CPU, fast)
        let tokens = self.sampler.sample_batch(&logits);

        // 3. Send to async pipeline (non-blocking!)
        for (req_id, token) in tokens {
            self.output_pipeline.sender.send(OutputEvent::Token {
                request_id: req_id, token_id: token, logprob: None,
            })?;
            // DON'T wait for detokenization. Next decode step starts immediately.
        }

        // 4. Immediately prepare next decode step
        self.prepare_next_step(&tokens);
        Ok(())
    }
}
```

**Gain:** Decode step latency reduced by 0.5-1ms (detokenize time no longer on critical path).

---

## 53. Recompute-Based Preemption

### 51.1 Problem

When a high-priority request arrives and GPU is full, we need to preempt a low-priority
request. Current options:
- **Swap KV to CPU** — works but expensive for long contexts (GBs of KV to transfer)
- **Kill and requeue** — wastes all work done so far

vLLM's third option: **Recompute** — drop KV, remember position, recompute from prompt when resumed.

### 51.2 When Recompute Beats Swap

```
Swap cost:  KV_size × (D2H_time + H2D_time_later)
Recompute cost: prompt_tokens × prefill_time_per_token

Recompute wins when: prompt is short relative to KV size accumulated.
Swap wins when: sequence has generated many tokens (KV large but prompt was short).
```

Break-even heuristic:
```rust
fn should_recompute(seq: &Sequence) -> bool {
    let swap_cost_ms = seq.kv_size_bytes() as f64 / PCIE_BANDWIDTH_BYTES_PER_MS;
    let recompute_cost_ms = seq.prompt_len() as f64 * PREFILL_MS_PER_TOKEN;
    recompute_cost_ms < swap_cost_ms
}
```

### 51.3 Design

```rust
pub enum PreemptionStrategy {
    /// Swap KV to CPU. Resume by copying back.
    Swap,
    /// Drop KV entirely. Resume by recomputing from prompt.
    Recompute,
    /// Choose automatically based on cost comparison.
    Auto,
}

pub struct PreemptedSequence {
    request_id: RequestId,
    strategy: PreemptionStrategy,
    /// For Swap: KV pages stored on CPU.
    swapped_pages: Option<Vec<CpuPage>>,
    /// For Recompute: just remember the prompt + generated tokens so far.
    checkpoint: SequenceCheckpoint,
}

pub struct SequenceCheckpoint {
    /// Original prompt tokens.
    prompt_tokens: Vec<TokenId>,
    /// Tokens generated before preemption (will regenerate same with deterministic sampling).
    generated_tokens: Vec<TokenId>,
    /// Position to resume generation from.
    resume_position: usize,
    /// Sampler state at preemption point.
    sampler_state: Option<SamplerCheckpoint>,
}

impl Scheduler {
    fn preempt(&mut self, victim: &mut Sequence, strategy: PreemptionStrategy) {
        match strategy {
            PreemptionStrategy::Swap => {
                // Copy KV pages to CPU
                let pages = self.kv_cache.swap_to_cpu(victim.id);
                self.preempted.push(PreemptedSequence {
                    swapped_pages: Some(pages),
                    checkpoint: victim.checkpoint(),
                    ..
                });
            }
            PreemptionStrategy::Recompute => {
                // Just drop KV. Keep checkpoint (prompt + generated tokens).
                self.kv_cache.free_all_pages(victim.id);
                self.preempted.push(PreemptedSequence {
                    swapped_pages: None,
                    checkpoint: victim.checkpoint(),
                    ..
                });
            }
            PreemptionStrategy::Auto => {
                let strat = if should_recompute(victim) {
                    PreemptionStrategy::Recompute
                } else {
                    PreemptionStrategy::Swap
                };
                self.preempt(victim, strat);
            }
        }
    }

    fn resume(&mut self, preempted: PreemptedSequence) {
        match preempted.strategy {
            PreemptionStrategy::Swap => {
                // Copy KV pages back from CPU
                self.kv_cache.restore_from_cpu(preempted.swapped_pages.unwrap());
            }
            PreemptionStrategy::Recompute => {
                // Re-run prefill on prompt + already-generated tokens
                let full_input = [&preempted.checkpoint.prompt_tokens[..],
                                  &preempted.checkpoint.generated_tokens[..]].concat();
                self.schedule_prefill(preempted.request_id, &full_input);
                // After prefill, resume decode from where we left off
            }
        }
    }
}
```

### 51.4 User Impact

Transparent. User sees slightly higher latency on resumed request (recompute time),
but the high-priority request got served immediately. No visible error or disconnect.

---

## 54. Frequency-Based Prefix Cache Eviction

### 52.1 Problem

LRU eviction for prefix cache is suboptimal:
- A prompt prefix used 100 times but not in the last 5 seconds gets evicted
- A one-off long prefix used once stays because it's recent

### 52.2 Design: LRU + Frequency Hybrid (LFU-like)

```rust
pub struct PrefixCacheEviction {
    policy: EvictionPolicy,
}

pub enum EvictionPolicy {
    /// Pure LRU (simple, current default).
    Lru,
    /// Pure frequency (most popular stay).
    Lfu,
    /// Hybrid: score = frequency_weight * frequency + recency_weight * recency.
    /// Similar to Redis's LFU or ARC (Adaptive Replacement Cache).
    Hybrid {
        frequency_weight: f32,  // default 0.6
        recency_weight: f32,    // default 0.4
        decay_period: Duration, // frequency decays over time (default 5 min)
    },
    /// Adaptive: auto-tune weights based on hit rate.
    Adaptive,
}

pub struct CachedPrefix {
    prefix_hash: u64,
    pages: Vec<PageId>,
    /// Access counter (with time decay).
    frequency: DecayingCounter,
    /// Last access time.
    last_access: Instant,
    /// Byte cost (longer prefix = more expensive to evict).
    size_pages: usize,
}

/// Counter that decays over time (avoids stale frequency from old bursts).
pub struct DecayingCounter {
    count: f32,
    last_decay: Instant,
    half_life: Duration,  // e.g. 5 minutes
}

impl DecayingCounter {
    fn increment(&mut self) {
        self.apply_decay();
        self.count += 1.0;
    }

    fn apply_decay(&mut self) {
        let elapsed = self.last_decay.elapsed();
        let decay_factor = 0.5_f32.powf(elapsed.as_secs_f32() / self.half_life.as_secs_f32());
        self.count *= decay_factor;
        self.last_decay = Instant::now();
    }

    fn value(&self) -> f32 {
        // Apply decay lazily on read
        let elapsed = self.last_decay.elapsed();
        let decay_factor = 0.5_f32.powf(elapsed.as_secs_f32() / self.half_life.as_secs_f32());
        self.count * decay_factor
    }
}

impl PrefixCacheEviction {
    fn eviction_score(&self, entry: &CachedPrefix) -> f64 {
        match &self.policy {
            EvictionPolicy::Hybrid { frequency_weight, recency_weight, .. } => {
                let freq_score = entry.frequency.value() as f64;
                let recency_score = 1.0 / (entry.last_access.elapsed().as_secs_f64() + 1.0);
                // Also consider cost: larger prefixes are more expensive to recompute
                let cost_factor = entry.size_pages as f64;

                (*frequency_weight as f64 * freq_score
                 + *recency_weight as f64 * recency_score)
                * cost_factor
            }
            _ => { /* LRU or LFU */ 0.0 }
        }
    }

    /// Evict entries with lowest score until target_pages freed.
    fn evict(&mut self, cache: &mut Vec<CachedPrefix>, target_pages: usize) -> Vec<PageId> {
        cache.sort_by(|a, b| self.eviction_score(a)
            .partial_cmp(&self.eviction_score(b)).unwrap());

        let mut freed = 0;
        let mut evicted_pages = vec![];
        while freed < target_pages && !cache.is_empty() {
            let victim = cache.remove(0);  // lowest score
            freed += victim.size_pages;
            evicted_pages.extend(victim.pages);
        }
        evicted_pages
    }
}
```

### 52.3 Adaptive Policy

```rust
/// Auto-tune frequency vs recency weights based on observed hit rate.
impl AdaptiveEviction {
    fn adapt(&mut self, hit: bool) {
        if hit {
            // Hit came from a frequent prefix → boost frequency weight
            if self.last_hit_was_frequent {
                self.frequency_weight = (self.frequency_weight + 0.01).min(0.9);
                self.recency_weight = 1.0 - self.frequency_weight;
            }
        } else {
            // Miss: recently evicted prefix was needed → boost recency weight
            self.recency_weight = (self.recency_weight + 0.01).min(0.9);
            self.frequency_weight = 1.0 - self.recency_weight;
        }
    }
}
```

### 52.4 User API

```python
session = nxrt.load("model.onnx", options={
    "cache.eviction_policy": "hybrid",      # lru | lfu | hybrid | adaptive
    "cache.frequency_weight": "0.6",
    "cache.recency_weight": "0.4",
    "cache.frequency_decay_minutes": "5",   # half-life for frequency counter
})
```

---

## 55. Weight Loading

### 53.1 Supported Formats

| Format | Priority | Notes |
|--------|----------|-------|
| **Safetensors** | Primary | mmap-friendly, zero-copy, header-indexed. Preferred for new models. |
| **ONNX external data** | Required | Standard ONNX pattern (model.onnx + model.onnx.data). Must support. |
| GGUF | Phase 5 | llama.cpp compat. Lower priority. |

### 53.2 Loading Strategy

```rust
pub enum WeightLoadStrategy {
    /// mmap entire file. OS handles page-in on demand.
    /// Best for: safetensors, large models, SSD-backed tiering.
    Mmap,
    /// Eager read into arena. Best for: small models, GPU-only.
    Eager,
    /// Lazy: only load tensor when first accessed (triggered by placement plan).
    /// Best for: offloaded models where not all weights go to GPU.
    Lazy,
}

impl WeightLoader {
    /// Load from safetensors (mmap, header gives tensor offsets).
    pub fn load_safetensors(path: &Path, strategy: WeightLoadStrategy) -> Result<WeightStore>;

    /// Load from ONNX external data (model.onnx.data or multiple .bin files).
    pub fn load_onnx_external(model_path: &Path, data_paths: &[PathBuf], strategy: WeightLoadStrategy) -> Result<WeightStore>;
}
```

### 53.3 Lazy Loading + Tiered Memory Integration

With lazy loading, weights start on SSD/disk. The paged memory system (§33) pages them
into RAM/VRAM on first access. Combined with placement plan: only weights assigned to
GPU tier get eagerly loaded; CPU/SSD-tier weights stay lazy until needed.

---

## 56. Minimal Build & Binary Size Control

### 54.1 Problem

Full runtime with all kernels, all EPs, tracer, and diagnostics = large binary.
Edge/mobile/WASM/embedded users need minimal builds. Server users want everything.

### 54.2 Design: Feature Flags + Kernel Selection

**Cargo features (compile-time):**

```toml
# crates/onnx-runtime-session/Cargo.toml
[features]
default = ["full"]

# Presets
full = ["kernels-all", "tracer", "cuda", "diagnostics"]
minimal = []  # bare minimum: IR + loader + CPU matmul + session API
server = ["full", "metrics", "lora"]
edge = ["kernels-transformer", "quantization"]

# Kernel groups
kernels-all = ["kernels-math", "kernels-attention", "kernels-normalization", "kernels-activation", "kernels-tensor", "kernels-quantization", "kernels-vision", "kernels-control"]
kernels-transformer = ["kernels-math", "kernels-attention", "kernels-normalization", "kernels-activation", "kernels-tensor", "kernels-quantization"]
kernels-math = []        # MatMul, Add, Mul, Gemm
kernels-attention = []   # GQA, MHA, RotaryEmbedding
kernels-normalization = [] # LayerNorm, RMSNorm, GroupNorm
kernels-activation = []  # Gelu, FastGelu, BiasGelu, Sigmoid, Relu
kernels-tensor = []      # Reshape, Transpose, Gather, Concat, Slice
kernels-quantization = [] # DequantizeLinear, MatMulNBits
kernels-vision = []      # Conv, Pool, Resize, BatchNorm
kernels-control = []     # If, Loop, Where

# Individual kernel (escape hatch for extreme minimal)
kernel-matmul = []
kernel-add = []
kernel-gqa = []
kernel-layernorm = []
# ... etc

# Components
tracer = ["dep:onnx-runtime-tracer"]
diagnostics = ["tracer"]  # auto-diagnosis needs tracer
metrics = []              # Prometheus/OTEL
cuda = ["dep:onnx-runtime-ep-cuda"]
lora = []
vmap = []                 # Auto-batching pass
```

**Example builds:**

```bash
# Full server build (everything)
cargo build --release  # default = full

# Minimal LLM inference (transformer kernels only, no vision, no tracer)
cargo build --release --no-default-features --features "kernels-transformer"

# Extreme minimal (only MatMul + Add + LayerNorm + Reshape — for a custom tiny model)
cargo build --release --no-default-features --features "kernel-matmul,kernel-add,kernel-layernorm,kernels-tensor"

# Edge device (transformer + quantization, no CUDA, no tracer)
cargo build --release --no-default-features --features "edge"

# WASM (no filesystem, no CUDA, minimal kernels)
cargo build --release --target wasm32-unknown-unknown --no-default-features --features "kernels-transformer"
```

### 54.3 Kernel Registry (Conditional Compilation)

```rust
// crates/onnx-runtime-ep-cpu/src/kernels/mod.rs

pub fn register_all(registry: &mut KernelRegistry) {
    #[cfg(feature = "kernels-math")]
    {
        registry.register(Box::new(MatMulKernel::new()));
        registry.register(Box::new(AddKernel::new()));
        registry.register(Box::new(MulKernel::new()));
        registry.register(Box::new(GemmKernel::new()));
    }

    #[cfg(feature = "kernels-attention")]
    {
        registry.register(Box::new(GroupQueryAttentionKernel::new()));
        registry.register(Box::new(MultiHeadAttentionKernel::new()));
        registry.register(Box::new(RotaryEmbeddingKernel::new()));
    }

    #[cfg(feature = "kernels-normalization")]
    {
        registry.register(Box::new(LayerNormKernel::new()));
        registry.register(Box::new(RmsNormKernel::new()));
    }

    // ... etc
}
```

**Missing kernel at runtime = clear error (not silent failure):**
```
❌ Unsupported operator: "Conv" (opset 22)

  This build was compiled without vision kernels.
  Add feature "kernels-vision" to enable Conv support.

  Build command: cargo build --features "kernels-vision"
  Or use the "full" preset: cargo build --features "full"
```

### 54.4 Model-Driven Minimal Build (CLI Tool)

Users shouldn't have to guess which features they need:

```bash
# Analyze model and output the minimal feature set
$ nxrt features model.onnx

Required features for model.onnx:
  kernels-math         (MatMul, Add, Mul, Sub, Div)
  kernels-attention    (GroupQueryAttention, RotaryEmbedding)
  kernels-normalization (SimplifiedLayerNormalization)
  kernels-activation   (FastGelu, BiasSplitGelu)
  kernels-tensor       (Reshape, Transpose, Gather, Concat)
  kernels-quantization (MatMulNBits, DequantizeLinear)

Minimal build command:
  cargo build --release --no-default-features \
    --features "kernels-math,kernels-attention,kernels-normalization,kernels-activation,kernels-tensor,kernels-quantization"

Estimated binary size: ~8 MB (vs ~45 MB full)
```

### 54.5 Binary Size Estimates

| Build | Estimated Size | Use Case |
|-------|---------------|----------|
| `full` (all kernels + CUDA + tracer + diagnostics) | ~45 MB | Server |
| `kernels-transformer` (LLM, no vision) | ~15 MB | LLM-only deployment |
| `edge` (transformer + quant, no tracer) | ~12 MB | Edge device |
| `minimal` + specific kernels | ~5-8 MB | Embedded / WASM |
| `minimal` only (IR + loader + session) | ~2 MB | Custom EP only (all compute delegated) |

### 54.6 Python Wheel Strategy

```bash
# Base package (CPU, transformer kernels)
pip install nxrt

# With CUDA
pip install nxrt[cuda]          # pulls onnx-runtime-ep-cuda wheel

# With specific EP
pip install nxrt[mlx]           # Apple Silicon
pip install nxrt[webgpu]        # WebGPU
pip install nxrt[qnn]           # Qualcomm

# Minimal (no optional deps)
pip install nxrt --no-deps
```

### 54.7 Design Choices

| Choice | Decision | Rationale |
|--------|----------|----------|
| Feature flags (not runtime config) | **Yes** | Dead code elimination at compile time. Actual binary size reduction. |
| Kernel groups (not individual by default) | **Yes** | Practical granularity. Individual kernels as escape hatch. |
| Model-driven feature detection | **Yes** | Users shouldn't guess. `nxrt features` tells them exactly what's needed. |
| Missing kernel = clear compile-like error | **Yes** | Not a silent wrong result. Tells you what feature to add. |
| Default = full | **Yes** | Most users want everything. Minimal is opt-in. |

---

## 57. EPContext Node — On-Disk Compiled-EP Interchange

### 55.1 What & Why

Compiled/hardware EPs (QNN, OpenVINO, TensorRT, Vitis AI, and eventually our own
CUDA EP) pay a large **convert + compile** cost the first time they see a
subgraph: parsing the ONNX partition, lowering it to the vendor IR, and invoking
the vendor AOT compiler (QNN graph prepare, OpenVINO blob compile, TRT engine
build). This can be **seconds to minutes** and often needs the vendor SDK/toolchain
present on the deployment box — unacceptable for edge/serverless startup and
awkward for shipping.

ORT solves this with the **`EPContext` node**: an ONNX contrib operator
(domain `com.microsoft`) that embeds a **pre-compiled EP binary directly inside
the ONNX model graph**. The generating run partitions the graph, compiles each
subgraph, and rewrites the model so each compiled partition becomes a single
`EPContext` node carrying the vendor blob (or a path to it). On subsequent loads
the EP recognizes the node and **loads the blob directly**, skipping
convert+compile entirely.

This is a first-class item for us because *the EP ecosystem is the moat* (§1):
being able to **consume** ORT-produced `*_ctx.onnx` models and **generate** them
in the exact ORT format is what lets our runtime slot into existing vendor
tooling and existing pre-compiled model artifacts without re-compilation.

> **Two forms of one thing.** §4 already defines an internal
> `EpContext { ep_name, ep_version, data, covered_nodes, device_fingerprint }`
> (the **runtime form** produced by `Ep::save_context()` / consumed by
> `Ep::load_context()`), and §41 defines a host-local **compilation cache**.
> The `EPContext` node defined here is the **on-disk / interchange form**: the
> serialized, in-graph, tool-portable representation of that same compiled
> context. §57.3 and §57.4 define the exact mapping between them.

### 55.2 Op Schema (ORT `com.microsoft::EPContext`)

Authoritative source: ORT [*EP Context Design*](https://onnxruntime.ai/docs/execution-providers/EP-Context-Design.html).

- **Op:** `EPContext`, **domain:** `com.microsoft`.
- **Inputs / outputs:** **variadic**. An `EPContext` node *replaces a partitioned
  subgraph*, so its inputs/outputs are exactly that subgraph's boundary tensors
  (in order). The runtime must not assume any fixed arity.
- **Attributes:**

| Attribute             | Type    | Default | Meaning                                                                                                                                                                     |
|-----------------------|---------|---------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `main_context`        | int64   | `1`     | `1` = this node references EP-context content holding the graph for **this** node. `0` = the graph comes from **another** node whose `main_context=1` (some EPs pack multiple graphs into one primary context blob).                     |
| `ep_cache_context`    | string  | —       | If `embed_mode=1`: the context **payload** (the compiled blob bytes). If `embed_mode=0`: the **path** to the context file, relative to the ONNX model file.                |
| `embed_mode`          | int64   | `1`     | `1` = `ep_cache_context` holds the payload inline. `0` = `ep_cache_context` holds an external file path.                                                                    |
| `ep_sdk_version`      | string  | opt     | SDK/toolchain version that generated the node (invalidation / diagnostics).                                                                                                |
| `onnx_model_filename` | string  | opt     | Original ONNX model filename.                                                                                                                                              |
| `hardware_architecture` | string | opt    | Target hardware architecture the blob was compiled for.                                                                                                                    |
| `partition_name`      | string  | opt     | Name of the ORT-partitioned graph this node represents.                                                                                                                    |
| `source`              | string  | opt     | **Unique EP-defined key** identifying which EP owns/consumes this node. ORT hosts multiple `EPContext` nodes for different EPs in one model and dispatches on this key. E.g. the QNN EP only accepts nodes with `source` = `QNN` / `QnnExecutionProvider`; the OpenVINO EP only accepts `source` = `OpenVINOExecutionProvider`. **Dispatch MUST key on this attribute — no hardcoded EP names in the runtime.** |
| `notes`               | string  | opt     | Free-form notes.                                                                                                                                                           |
| `max_size`            | int64   | `0`     | Optional size hint for the context payload.                                                                                                                                |

- **Generation session options** (see §21.4 for the C-API surface):
  - `ep.context_enable` (int, `1` dumps a context-cache model),
  - `ep.context_file_path` (default `<orig>_ctx.onnx`),
  - `ep.context_embed_mode` (int, embed vs external).

### 55.3 Consuming (Load Path)

**Loader recognition & IR representation (`onnx-runtime-loader`).** The protobuf
parser (§19) already builds a `Node` per `NodeProto` with `op_type`, `domain`, and
attributes. An `EPContext` node is *just a node* with `domain =
"com.microsoft"`, `op_type = "EPContext"`, variadic i/o, and the attributes
above — no special IR node kind is required. The loader adds a typed *view* over
that node so downstream crates don't re-parse attributes, and resolves the blob
source at load time:

```rust
/// Typed view over a com.microsoft::EPContext node in the Graph IR.
/// Backed by the ordinary Node + its attributes — no separate IR node kind.
pub struct EpContextNode<'g> {
    pub node: NodeId,
    pub source: Option<&'g str>,      // `source` attr — the dispatch key (§57.6)
    pub main_context: bool,           // main_context != 0
    pub embed_mode: EmbedMode,        // Embedded | ExternalFile
    pub sdk_version: Option<&'g str>,
    pub partition_name: Option<&'g str>,
    // variadic boundary tensors come straight from node.inputs / node.outputs
}

pub enum EmbedMode { Embedded, ExternalFile }

/// Where the compiled blob physically lives after load-time resolution.
pub enum EpContextBlob {
    /// embed_mode=1: bytes owned inline (copied out of ep_cache_context).
    Embedded(Vec<u8>),
    /// embed_mode=0: mmap of an external file resolved *relative to the model dir*.
    External { path: PathBuf, map: Mmap },
}

impl Loader {
    /// Resolve the payload for one EPContext node. For embed_mode=0 the path in
    /// `ep_cache_context` is joined onto the model directory (same rule as
    /// external initializer data, §19.2) and mmap'd read-only.
    fn resolve_ep_context(&self, model_dir: &Path, n: &EpContextNode) -> Result<EpContextBlob>;
}
```

Key loader rules:

- `embed_mode=0` paths are resolved **relative to the ONNX model file** (identical
  policy to external-weight resolution, §19.2) and **mmap'd / read lazily** — the
  blob is never eagerly copied into the graph.
- The loader does **not** interpret the blob; it is opaque vendor bytes.
- Shape inference (§19.3) treats `EPContext` as opaque: output shapes come from
  the model's `value_info` / graph outputs, not from op-specific inference.

**Dispatch & execution (`onnx-runtime-session` + `onnx-runtime-ep-api`).** During
placement, an `EPContext` node bypasses the cost model (like `claim_nodes`, §4.1):
it is handed to the EP whose declared `source` key matches the node's `source`
attribute, via a **`source`-keyed registry** (§57.6). That EP turns the on-disk
node into the internal runtime form and restores it:

```rust
// The on-disk node → internal EpContext (§4) → EP restore.
fn dispatch_ep_context(reg: &EpContextRegistry, node: &EpContextNode, blob: EpContextBlob)
    -> Result<()>
{
    let ep = reg.claim(node.source)          // §57.6 — match on `source`, never a hardcoded name
        .ok_or(Error::NoEpForContext { source: node.source.map(str::to_owned) })?;

    // Serialized (on-disk) form → runtime form expected by Ep::load_context (§4).
    let ctx = EpContext {
        ep_name: ep.name().to_string(),
        ep_version: node.sdk_version.unwrap_or_default().to_string(), // ← ep_sdk_version
        data: match blob {                                            // ← ep_cache_context
            EpContextBlob::Embedded(b) => b,
            EpContextBlob::External { map, .. } => map[..].to_vec(),
        },
        covered_nodes: vec![node.node],       // this node's boundary == the partition
        device_fingerprint: String::new(),    // filled/validated by the EP
    };
    ep.load_context(&ctx)                      // EP skips convert+compile
}
```

**`main_context` multi-graph referencing.** Some EPs (notably QNN) pack multiple
compiled graphs into one primary context binary. Nodes with `main_context=1`
*own* the payload; nodes with `main_context=0` **reference** a sibling primary
node's already-loaded context (matched by `source` + `partition_name`). The
session loads all `main_context=1` blobs first (deduplicating identical
`ep_cache_context` payloads), then resolves `main_context=0` nodes against the
context the owning EP already holds — no second blob load.

### 55.4 Generating (Dump Path)

When `ep.context_enable` is set (§21.4), after the session partitions the graph and
each EP compiles its claimed subgraph, the session asks every participating EP for
its compiled context and **serializes it back into `EPContext` nodes** in a new
`*_ctx.onnx` model:

```
compile subgraphs → for each compiled partition:
    ctx: EpContext = ep.save_context()?          // §4 runtime form
    build EPContext NodeProto:
        domain     = "com.microsoft"
        op_type    = "EPContext"
        input/output = partition boundary tensors (variadic, in order)
        source            = ep.context_source_key()   // EP's own `source` key (§57.6)
        ep_sdk_version    = ctx.ep_version
        partition_name    = <partition id>
        main_context      = 1
        embed_mode        = options.embed_mode
        ep_cache_context  = match embed_mode {
            Embedded     => ctx.data (inline bytes),
            ExternalFile => write ctx.data to "<model_stem>_<source>_<part>.bin"
                            next to the ctx model; store the RELATIVE path string,
        }
→ replace the compiled subgraph in the ModelProto with the EPContext node
→ serialize to options.file_path (default "<orig>_ctx.onnx")
```

Ownership: the **writer lives in `onnx-runtime-loader`** (it already owns
ONNX↔IR protobuf serialization and external-data file I/O), driven by
`onnx-runtime-session` (which owns compilation, partition boundaries, and the
`ep.context_*` options). For `embed_mode=0`, the loader writes the external
binary next to the ctx model and stores the **relative** path in
`ep_cache_context`, mirroring the §19.2 external-data convention so the produced
model round-trips through §57.3's load path (and through upstream ORT).

### 55.5 C-API Surface

The `ep.context_enable` / `ep.context_file_path` / `ep.context_embed_mode`
session options are exposed through the ORT-compatible C API so existing ORT
tooling that dumps context-cache models works unchanged against
`libonnxruntime.so`. See **§21.4** for the key table and the parsed
`EpContextGenOptions` struct.

### 55.6 Model-Agnostic Dispatch (Hard Rule)

EP selection for an `EPContext` node is **always** by the node's `source`
attribute, resolved through a registry — **never** by hardcoded EP names or
string-matching vendor identifiers in the runtime. Each EP declares the
`source` key(s) it accepts; the registry maps key → EP. This keeps the runtime
generalizable and config-driven: adding a new compiled EP requires no change to
loader/session dispatch code.

```rust
/// Registry mapping an EPContext `source` key → the EP that owns it.
/// EPs register the key(s) they accept; dispatch is a pure lookup.
pub struct EpContextRegistry {
    by_source: HashMap<String, EpId>,   // e.g. "QNN"/"QnnExecutionProvider" → qnn ep
}

impl EpContextRegistry {
    /// EP declares which `source` key(s) it consumes (from EP config, not code).
    pub fn register(&mut self, ep: EpId, source_keys: &[String]);
    /// Look up the EP for a node's `source` attribute. None ⇒ node unclaimed.
    pub fn claim(&self, source: Option<&str>) -> Option<EpId>;
}
```

An `EPContext` node whose `source` matches no registered EP is **unclaimed**: the
session surfaces a clear error (`NoEpForContext { source }`) rather than guessing —
the model requires an EP that is not loaded.

### 55.7 Crate Responsibilities & Phasing

| Crate                    | Responsibility                                                                                             |
|--------------------------|-----------------------------------------------------------------------------------------------------------|
| `onnx-runtime-loader`    | Recognize `com.microsoft::EPContext` nodes; `EpContextNode` view; resolve `embed_mode=0` paths (rel. to model) + mmap; **writer** for the `*_ctx.onnx` dump path (§57.4) incl. external-blob files. |
| `onnx-runtime-ep-api`    | `EpContextRegistry` (`source`-keyed); the on-disk-node ↔ internal `EpContext` mapping contract; EP declares accepted `source` keys + `save_context()`/`load_context()` (already in §4). |
| `onnx-runtime-session`   | Bypass placement for `EPContext` nodes; drive `main_context=1/0` resolution + dedup; own `ep.context_*` options and drive the dump path during compile. |
| `onnx-runtime-capi`      | Parse `ep.context_enable`/`ep.context_file_path`/`ep.context_embed_mode` session options (§21.4).         |

**Roadmap placement: Phase 2 (§56)**, alongside *Legacy EP loading (dlopen +
vtable)* and the *ORT Graph ABI bridge* — `EPContext` is precisely the mechanism
by which compiled/legacy EPs interoperate with our runtime. It is **foundational
for the CUDA/QNN/OpenVINO/TensorRT EPs** even though **no Phase-1 EP consumes it**
(the pure-Rust CPU EP has no compile step). Building the loader recognition,
`source`-keyed dispatch, and the dump/writer path in Phase 2 means the first
compiled EP lands into an already-working context-cache pipeline.

---

## 58. Phased Roadmap

### Phase 1: Foundation (8-12 weeks)
- [ ] `onnx-runtime-ir`: Graph IR with all types, validation, mutation API
- [ ] `onnx-runtime-loader`: ONNX protobuf parser, shape inference, weight mmap (safetensors + external data)
- [ ] `onnx-runtime-ep-api`: ExecutionProvider trait, Kernel trait, OpRegistry
- [ ] `onnx-runtime-ep-cpu`: Basic ops (MatMul, Add, Relu, Reshape, Transpose, Gather, LayerNorm) via oneDNN
- [ ] `onnx-runtime-session`: SessionBuilder, sequential executor (no async), basic Run API
- [ ] `onnx-runtime-capi`: OrtGetApiBase + CreateSession + Run (Tier 1 C API)
- [ ] **Milestone: run BERT on CPU, output matches ORT**

### Phase 2: Multi-Device + EPs (8-12 weeks)
_Depends on: Phase 1 complete_
- [ ] `onnx-runtime-ep-cuda`: CUDA EP on the **§15 kernel stack** — `cudarc` foundation, cuBLASLt GEMM (fused epilogue), CuTe custom kernels (LayerNorm/RMSNorm/RoPE/softmax) via `extern "C"` FFI, cuDNN fused SDPA for attention (Phase 2a). _cuTile deferred to Phase 3 (no Hopper/SM90, Python-only — §15.8)._
- [ ] `onnx-runtime-ep-api`: ORT Graph ABI bridge for legacy plugin EPs
- [ ] Legacy EP loading (dlopen + vtable)
- [ ] **EPContext node support (§57): loader recognition + IR representation, `source`-keyed EP dispatch (ep-api/session), embed/external blob resolution, and the `*_ctx.onnx` writer for `ep.context_enable` (loader/session), plus `ep.context_*` C-API session options (§21.4).** Foundational for the compiled EPs (CUDA/QNN/OpenVINO/TensorRT); no Phase-1 EP consumes it yet.
- [ ] `onnx-runtime-cost-model`: Static cost formulas + device profiles
- [ ] `onnx-runtime-optimizer`: ConstantFolding, DeadNodeElimination, OpFusion, AttentionFusion
- [ ] Layout propagation pass
- [ ] Placement optimizer (greedy first, ILP stretch goal)
- [ ] `onnx-runtime-scheduler`: Async DAG executor with streams + fences
- [ ] Transfer insertion pass + async transfer
- [ ] **Milestone: run Llama on CUDA EP, 2+ EPs in one graph**

### Phase 3: Performance (6-10 weeks)
_Depends on: Phase 2 complete_
- [ ] `onnx-runtime-memory`: Arena allocator, lifetime analysis, buffer aliasing, in-place detection
- [ ] Double-buffered async transfers
- [ ] CUDA graph capture for decode step
- [ ] Compute-communication overlap (micro-chunking)
- [ ] FlashAttention integration (flash-attn library binding)
- [ ] CuTe kernels: FusedResidualLayerNorm, RoPE, FusedGEMM+Bias+Act
- [ ] `onnx-runtime-tracer`: Unified tracing crate (TraceContext, collectors, Perfetto/Chrome/JSONL, auto-diagnosis)
- [ ] Cost model calibration + profiling feedback loop
- [ ] Dynamic shape kernel cache + bucketing
- [ ] ILP placement optimizer (HiGHS integration)
- [ ] **Milestone: match or beat ORT latency on Llama decode**

### Phase 4: GenAI Integration (4-8 weeks)
_Depends on: Phase 3 compute kernels working_
- [ ] `backend-nxrt` feature flag in onnx-genai-engine
- [ ] KV cache on onnx-runtime-memory arenas
- [ ] Continuous batching through onnx-runtime-scheduler
- [ ] Paged FlashAttention with block table
- [ ] End-to-end: ONNX model → GenAI server, zero ORT dependency
- [ ] **Milestone: onnx-genai-server runs Llama with backend-nxrt**

### Phase 5: Ecosystem (ongoing)
_Depends on: Phase 2 C API working_
- [ ] Python bindings (`nxrt` + per-EP packages)
- [ ] `onnx-runtime-autotuner`: Agent-driven optimization loop
- [ ] More EP crates: mlx, coreml, webgpu, qnn, openvino, rocm
- [ ] GGUF weight loading
- [ ] Conformance test suite (top 50 HuggingFace models)
- [ ] Benchmark CI + regression tracking
- [ ] ORT C API Tier 2 + Tier 3 functions
- [ ] Documentation + EP development guide
- [ ] **Milestone: Python onnxruntime drop-in replacement works**
