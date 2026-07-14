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
10. [Compute-Communication Overlap](#10-compute-communication-overlap)
11. [Dynamic Shape Specialization](#11-dynamic-shape-specialization)
12. [Weight Loading and Storage](#12-weight-loading-and-storage)
13. [Flash Attention Integration](#13-flash-attention-integration)
14. [CUDA Graph Capture](#14-cuda-graph-capture)
15. [CuTe Kernel Strategy](#15-cute-kernel-strategy)
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

## 15. CuTe Kernel Strategy

### 15.1 What is CuTe

CuTe (CUDA Tile) is CUTLASS 3.x's core abstraction. It models:
- **Layouts** as algebraic objects (compose, divide, complement)
- **Tiling** as layout transformations
- **Data movement** (shared memory staging, register tiling, TMA) as layout operations

### 15.2 When to Use CuTe vs cuBLAS/cuDNN

| Op | Strategy | Reason |
|----|----------|--------|
| GEMM (standard shapes) | cuBLAS | Battle-tested, auto-tuned |
| Fused GEMM+Bias+Act | CuTe | Custom fusion, cuBLAS can't fuse |
| FlashAttention | flash-attn library | Specialized, heavily optimized |
| LayerNorm | CuTe | Simple enough, good learning exercise |
| RoPE | CuTe | Element-wise with position-dependent pattern |
| Quantized MatMul (INT4×FP16) | CuTe | Custom dequant+GEMM fusion |
| Residual+LayerNorm fusion | CuTe | Cross-op fusion not in cuDNN |

### 15.3 CuTe Kernel Example: Fused Residual + LayerNorm

```cpp
// native-eps/cuda/src/kernels/fused_residual_layernorm.cu
#include <cute/tensor.hpp>

template<int kBlockSize = 1024>
__global__ void fused_residual_layernorm(
    float const* residual,   // [batch, seq, hidden]
    float const* input,      // [batch, seq, hidden]
    float const* gamma,      // [hidden]
    float const* beta,       // [hidden]
    float* output,           // [batch, seq, hidden]
    int hidden_size,
    float eps
) {
    using namespace cute;
    // Each thread block handles one (batch, seq) position
    int idx = blockIdx.x;

    // CuTe layout for the hidden dimension
    auto layout = make_layout(make_shape(hidden_size));

    // Load residual + input (fused add)
    // Compute mean and variance in registers
    // Normalize and apply gamma/beta
    // Write output
    // All in one kernel — no intermediate buffer for residual add
}
```

### 15.4 Hopper-Specific: TMA + WGMMA

For SM90 (H100/H200), CuTe provides direct access to:
- **TMA** (Tensor Memory Accelerator): async global→shared copy without using warps
- **WGMMA** (Warpgroup Matrix Multiply-Accumulate): fused shared-memory GEMM

```cpp
// Hopper TMA example via CuTe
auto tma_load = make_tma_copy(SM90_TMA_LOAD{}, source_tensor, smem_layout);
// Issues async load from global → shared, freeing warps for compute
cute::copy(tma_load, source_tensor, shared_tensor);
```

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
ort2 profile model.onnx --inputs input.npz --runs 100 --output trace.json

# Compare two runs
ort2 compare trace_before.json trace_after.json

# Dump graph at each optimization stage
ort2 dump-passes model.onnx --format dot --output-dir passes/

# Memory analysis
ort2 memory model.onnx --inputs input.npz --output memory_report.json

# Validate (check output matches ORT)
ort2 validate model.onnx --inputs input.npz --reference-output ort_output.npz --tolerance 1e-5
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
session = ort2.load("model.onnx")
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
session = ort2.load("model.onnx", hints={
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
import ort2

# Zero-config:
session = ort2.load("model.onnx")
output = session.run(input_ids=input_array)

# Device preference:
session = ort2.load("model.onnx", device="gpu")

# Namespaced options:
session = ort2.load("model.onnx",
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
│   ├── onnx-runtime-ep-cuda/                      # CUDA EP (CuTe + cuBLAS) — we maintain
│   ├── onnx-runtime-session/                      # Session builder, inference API
│   ├── onnx-runtime-profiler/                     # Tracing, Chrome Trace, memory debugger
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
│   └── python/                           # PyO3: ort2 + per-EP packages
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
backend-ort2 = ["dep:onnx-runtime-session"]        # our runtime
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

### 24.1 Main Package: `ort2`

```python
import ort2
from ort_ep_cuda import CudaEp

# Create session with our runtime
session = ort2.InferenceSession(
    "model.onnx",
    providers=[CudaEp(device_id=0), ort2.CpuEp()],
    optimization_level="aggressive",
)

# Run inference
outputs = session.run({"input_ids": input_array})

# Profile
report = session.profile({"input_ids": input_array}, num_runs=10)
print(report.bottlenecks)

# Auto-tune
tuner = ort2.AutoTuner(session)
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

| Platform | EP crates available | Weight mmap | CUDA Graph | Notes |
|----------|-------------------|-------------|------------|-------|
| Linux x64 | cpu, cuda, rocm, openvino, qnn, webgpu | ✅ | ✅ | Primary |
| macOS arm64 | cpu, mlx, coreml, webgpu | ✅ | ❌ | MLX for GPU |
| Windows x64 | cpu, cuda, openvino, webgpu | ✅ | ✅ | |
| Linux arm64 | cpu, qnn, webgpu | ✅ | ❌ | Edge |
| Web (WASM) | webgpu | ❌ (fetch) | ❌ | wasm-bindgen |

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
    let our_output = run_with_ort2(model_path, inputs)?;
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
session = ort2.load("llama-70b.onnx")  # 16GB GPU? Fine. Auto-offloads.

# Explicit memory limit (leave room for other processes)
session = ort2.load("model.onnx", memory_limit=12 * GB)

# Force full offload (CPU-only mode)
session = ort2.load("model.onnx", device="cpu")
```

Namespaced options for fine control:
```python
session = ort2.load("model.onnx", options={
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
import ort2

# Auto: single model, best available device(s)
session = ort2.load("model.onnx")

# Tensor parallelism (split across 4 GPUs)
session = ort2.load("llama-70b.onnx", options={
    "parallel.strategy": "tensor",
    "parallel.degree": "4",
})

# Pipeline parallelism (explicit layer→GPU mapping)
session = ort2.load("llama-70b.onnx", options={
    "parallel.strategy": "pipeline",
    "parallel.stages": "0-15:gpu:0,16-31:gpu:1",
})

# Data parallelism (throughput mode for serving)
session = ort2.load("model.onnx", options={
    "parallel.strategy": "data",
    "parallel.replicas": "4",
})

# Multi-session priority management
chat = ort2.load("llama.onnx", priority="realtime", options={"memory.gpu_budget_mb": "10000"})
image = ort2.load("sdxl.onnx", priority="background", options={"memory.gpu_budget_mb": "6000"})
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

## 32. Phased Roadmap

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
- [ ] `onnx-runtime-ep-cuda`: CUDA EP with cuBLAS GEMM + CuTe LayerNorm/GELU
- [ ] `onnx-runtime-ep-api`: ORT Graph ABI bridge for legacy plugin EPs
- [ ] Legacy EP loading (dlopen + vtable)
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
- [ ] `onnx-runtime-profiler`: Chrome Trace + Perfetto export, cross-device timeline
- [ ] Cost model calibration + profiling feedback loop
- [ ] Dynamic shape kernel cache + bucketing
- [ ] ILP placement optimizer (HiGHS integration)
- [ ] **Milestone: match or beat ORT latency on Llama decode**

### Phase 4: GenAI Integration (4-8 weeks)
_Depends on: Phase 3 compute kernels working_
- [ ] `backend-ort2` feature flag in onnx-genai-engine
- [ ] KV cache on onnx-runtime-memory arenas
- [ ] Continuous batching through onnx-runtime-scheduler
- [ ] Paged FlashAttention with block table
- [ ] End-to-end: ONNX model → GenAI server, zero ORT dependency
- [ ] **Milestone: onnx-genai-server runs Llama with backend-ort2**

### Phase 5: Ecosystem (ongoing)
_Depends on: Phase 2 C API working_
- [ ] Python bindings (`ort2` + per-EP packages)
- [ ] `onnx-runtime-autotuner`: Agent-driven optimization loop
- [ ] More EP crates: mlx, coreml, webgpu, qnn, openvino, rocm
- [ ] GGUF weight loading
- [ ] Conformance test suite (top 50 HuggingFace models)
- [ ] Benchmark CI + regression tracking
- [ ] ORT C API Tier 2 + Tier 3 functions
- [ ] Documentation + EP development guide
- [ ] **Milestone: Python onnxruntime drop-in replacement works**
