# Eager Execution — Single Op Dispatch (Python Binding)

> Runtime-level eager execution for nxrt. Dispatch individual ONNX ops to EP kernels
> without building a graph. PyTorch-style experience with ONNX op semantics.

**Scope:** Python API design, Rust implementation, opset versioning, custom domains, subgraph ops.
Complements [ORT2.md](./ORT2.md) §20 (Session API) and §24 (Python Bindings).

---

## Table of Contents

1. [Design Principles](#1-design-principles)
2. [Python API](#2-python-api)
3. [Tensor](#3-tensor)
4. [Op Dispatch](#4-op-dispatch)
5. [Opset Versioning](#5-opset-versioning)
6. [Custom Domains](#6-custom-domains)
7. [Subgraph Ops](#7-subgraph-ops)
8. [Kernel Registry & Cache](#8-kernel-registry--cache)
9. [Shape Inference](#9-shape-inference)
10. [Rust Implementation](#10-rust-implementation)
11. [PyO3 Binding Layer](#11-pyo3-binding-layer)
12. [Crate & Directory Structure](#12-crate--directory-structure)
13. [Design Decisions](#13-design-decisions)

---

## 1. Design Principles

1. **`ops.*` = Pythonic, latest opset.** Convenience functions with clean signatures. Users
   don't need to know what opset version they're using.

2. **`dispatch()` = ONNX-native escape hatch.** Any op, any domain, any opset. 1:1 mapping
   to ONNX op semantics.

3. **Opset versioning is explicit, never surprising.** Default is latest. User can pin
   globally, locally (context manager), or per-call. We never auto-break `ops.*` signatures.

4. **Custom domains are first-class.** `com.microsoft`, `ai.onnx.ml`, user-registered domains
   all go through the same dispatch path with independent opset versions.

5. **Subgraph ops use Python callables.** Eager mode's whole point is Python-native control
   flow. If/Loop/Scan bodies are lambdas, not serialized ONNX subgraphs.

6. **Device mismatch is an error.** No implicit cross-device transfers. User must explicitly
   `.to(device)`. Silent transfers are performance traps.

7. **Strided tensors are first-class.** Transpose returns a view (O(1)). Kernels that need
   contiguous input auto-contiguize. Matches ORT2.md §5.

---

## 2. Python API

### 2.1 Quick Start

```python
import nxrt

# Create tensors
a = nxrt.tensor([1.0, 2.0, 3.0], device="cuda:0")
b = nxrt.tensor([4.0, 5.0, 6.0], device="cuda:0")

# Dispatch ops — Pythonic API
out = nxrt.ops.add(a, b)
out = nxrt.ops.matmul(a.reshape(1, 3), b.reshape(3, 1))
out = nxrt.ops.relu(out)

# Generic dispatch — any ONNX op
out = nxrt.dispatch("MatMul", inputs=[a, b])
```

### 2.2 Device Context

```python
# Default device (avoid passing device= every time)
with nxrt.device("cuda:0"):
    x = nxrt.tensor([1.0, 2.0])    # automatically on cuda:0
    y = nxrt.ops.relu(x)            # dispatches to CUDA EP

# Explicit transfer
x_gpu = x.to("cuda:0")
x_cpu = x_gpu.to("cpu")
```

---

## 3. Tensor

### 3.1 Creation

```python
# From Python list / scalar
a = nxrt.tensor([1.0, 2.0, 3.0], device="cuda:0", dtype=nxrt.float32)
s = nxrt.tensor(42, dtype=nxrt.int64)

# From numpy (copies to device)
a = nxrt.tensor(numpy_array, device="cuda:0")

# Zero-copy interop via DLPack
a = nxrt.from_dlpack(torch_tensor)    # torch → nxrt, zero-copy
t = torch.from_dlpack(a)              # nxrt → torch, zero-copy

# Zeros / ones / empty
a = nxrt.zeros((3, 4), device="cuda:0", dtype=nxrt.float16)
a = nxrt.ones((3, 4), device="cuda:0")
a = nxrt.empty((3, 4), device="cuda:0")  # uninitialized
```

### 3.2 Properties & Conversion

```python
a.shape       # (3,)
a.dtype       # nxrt.float32
a.device      # DeviceId("cuda", 0)
a.strides     # (1,) — physical strides exposed (ORT2.md §5)

a.numpy()     # → numpy array (D2H copy if on GPU)
a.dlpack()    # → DLManagedTensor (zero-copy export)

a.is_contiguous()  # True
a.contiguous()     # copy to contiguous layout if needed
a.to("cpu")        # explicit device transfer
```

### 3.3 Strided Tensor Support

```python
# Transpose returns a view — O(1), no data copy
t = nxrt.ops.transpose(a, perm=[1, 0])
t.is_contiguous()  # False
t.strides          # permuted strides

# Kernels that support strided input accept it directly.
# Kernels that don't → auto-contiguize (documented per-op).
out = nxrt.ops.matmul(t, b)  # matmul may auto-contiguize t
```

### 3.4 Rust: PyTensor

```rust
#[pyclass]
pub struct PyTensor {
    inner: Tensor,
}

#[pymethods]
impl PyTensor {
    #[new]
    #[pyo3(signature = (data, /, device=None, dtype=None))]
    fn new(data: &Bound<'_, PyAny>, device: Option<&str>, dtype: Option<&str>) -> PyResult<Self>;

    #[staticmethod]
    fn from_dlpack(obj: &Bound<'_, PyAny>) -> PyResult<Self>;

    fn __dlpack__(&self, py: Python<'_>) -> PyResult<PyObject>;
    fn __dlpack_device__(&self) -> (i32, i32);

    fn numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDyn<f32>>>;

    fn to(&self, device: &str) -> PyResult<Self>;
    fn contiguous(&self) -> PyResult<Self>;
    fn is_contiguous(&self) -> bool;

    #[getter] fn shape(&self) -> Vec<usize>;
    #[getter] fn dtype(&self) -> &str;
    #[getter] fn device(&self) -> String;
    #[getter] fn strides(&self) -> Vec<i64>;

    fn __repr__(&self) -> String;
    fn __str__(&self) -> String;
}
```

---

## 4. Op Dispatch

### 4.1 `nxrt.ops.*` — Typed Convenience Functions

```python
# Math
out = nxrt.ops.add(a, b)
out = nxrt.ops.sub(a, b)
out = nxrt.ops.mul(a, b)
out = nxrt.ops.div(a, b)
out = nxrt.ops.matmul(a, b)
out = nxrt.ops.gemm(a, b, c, alpha=1.0, beta=1.0, transA=False, transB=False)

# Activation
out = nxrt.ops.relu(x)
out = nxrt.ops.sigmoid(x)
out = nxrt.ops.tanh(x)
out = nxrt.ops.gelu(x)
out = nxrt.ops.silu(x)

# Normalization
out = nxrt.ops.layer_norm(x, gamma, beta, axis=-1, epsilon=1e-5)
out = nxrt.ops.batch_norm(x, scale, bias, mean, var, epsilon=1e-5, momentum=0.9)

# Reduction
out = nxrt.ops.reduce_mean(x, axes=[1], keepdims=True)
out = nxrt.ops.reduce_sum(x, axes=[1], keepdims=True)

# Shape manipulation
out = nxrt.ops.reshape(x, shape)          # shape is a tensor or list
out = nxrt.ops.transpose(x, perm=[1, 0])  # returns view (O(1))
out = nxrt.ops.concat([a, b, c], axis=0)
out = nxrt.ops.split(x, num_outputs=3, axis=0)
out = nxrt.ops.squeeze(x, axes=[1])
out = nxrt.ops.unsqueeze(x, axes=[0])

# Other
out = nxrt.ops.softmax(x, axis=-1)
out = nxrt.ops.cast(x, to=nxrt.float16)
values, indices = nxrt.ops.topk(x, k=5, axis=-1)
out = nxrt.ops.gather(x, indices, axis=0)
out = nxrt.ops.where_(condition, x, y)
```

**Signature contract:** `ops.*` signatures follow the latest ONNX opset at the time of the
nxrt major version release. They do not auto-change when ONNX publishes a new opset.
See [§5 Opset Versioning](#5-opset-versioning).

### 4.2 `nxrt.dispatch()` — Generic Op Dispatch

```python
out = nxrt.dispatch(
    "MatMul",                     # op_type (required)
    inputs=[a, b],                # positional inputs (required)
    # --- optional ---
    domain="",                    # ONNX domain (default: standard "")
    opset=26,                     # opset version (default: context opset)
    # All other kwargs → ONNX attributes
    alpha=0.5,                    # attribute
)

# Multi-output: returns tuple
values, indices = nxrt.dispatch("TopK", inputs=[x], k=5, axis=-1)
```

**Return convention:**
- Single output → `PyTensor`
- Multiple outputs → `tuple[PyTensor, ...]`

### 4.3 EP Selection (Automatic)

Eager dispatch resolves EP from input tensors:

1. All inputs on same device → dispatch to that device's EP
2. Mixed devices → **error** (user must `.to(device)` explicitly)
3. No inputs (e.g. Constant) → use default device
4. EP doesn't have kernel for this op → fallback to CPU EP (with explicit warning)

```python
# All on CUDA → CUDA EP handles it
a = nxrt.tensor([1.0], device="cuda:0")
b = nxrt.tensor([2.0], device="cuda:0")
out = nxrt.ops.add(a, b)  # → CUDA EP

# Mixed → error
c = nxrt.tensor([3.0], device="cpu")
out = nxrt.ops.add(a, c)  # Error: mixed devices {cuda:0, cpu}
```

---

## 5. Opset Versioning

### 5.1 Problem

ONNX opsets introduce breaking changes: parameters move from attributes to inputs,
default values change, new required parameters appear. Users need control without
surprises.

### 5.2 Three Levels of Control

```python
import nxrt

# Level 1: Global default (process-wide)
nxrt.default_opset           # 26 (latest at nxrt release)
nxrt.default_opset = 18      # pin everything to opset 18

# Level 2: Context manager (block-scoped)
with nxrt.opset(18):
    out = nxrt.ops.resize(x, sizes=target)  # opset 18 semantics

# Level 3: Per-call (finest granularity, dispatch() only)
out = nxrt.dispatch("Resize", [x, roi, scales, sizes], opset=11)
```

**Priority:** `dispatch(opset=N)` > `with nxrt.opset(N)` > `nxrt.default_opset`

### 5.3 Context Manager Implementation

```python
class opset:
    """Context manager for opset version override.

    Supports standard domain + custom domains via keyword arguments.

    Examples:
        with nxrt.opset(18):                          # standard domain = 18
        with nxrt.opset(microsoft=2):                  # com.microsoft = 2
        with nxrt.opset(18, microsoft=2, ml=3):        # multiple domains
    """

    # Convenient aliases for common domains
    DOMAIN_ALIASES = {
        "microsoft": "com.microsoft",
        "ml": "ai.onnx.ml",
        "training": "ai.onnx.training",
        "preview": "ai.onnx.preview.training",
    }

    def __init__(self, version: int | None = None, **domains: int):
        self.overrides: dict[str, int] = {}
        if version is not None:
            self.overrides[""] = version  # standard ONNX domain
        for alias, ver in domains.items():
            domain = self.DOMAIN_ALIASES.get(alias, alias)
            self.overrides[domain] = ver

    def __enter__(self):
        self._prev = _get_thread_opset().copy()
        _get_thread_opset().update(self.overrides)
        return self

    def __exit__(self, *exc):
        _set_thread_opset(self._prev)
```

### 5.4 `ops.*` Versioning Contract

`ops.*` functions expose the **latest opset signature at the time of the nxrt major version**.
They do not auto-change across nxrt minor/patch versions.

```python
# If a new ONNX opset changes Resize's API:
# - nxrt 1.x: ops.resize() keeps opset-18 signature
# - nxrt 2.0: may update ops.resize() signature (semver major bump)
# - Users who need the new opset before 2.0: use dispatch()
```

When `ops.*` is called under a downgraded opset context that makes a parameter invalid:

```python
with nxrt.opset(10):
    # coordinate_transform_mode doesn't exist in opset 10
    nxrt.ops.resize(x, sizes=s, coordinate_transform_mode="half_pixel")
    # → Error: 'coordinate_transform_mode' requires opset >= 11 (current context: 10)
```

### 5.5 Rust: Opset Resolution

```rust
/// Per-domain opset version resolution.
pub struct OpsetContext {
    /// domain → opset version
    versions: HashMap<String, u64>,
}

impl OpsetContext {
    /// Resolve effective opset for a domain.
    ///
    /// Priority: explicit > thread-local (context manager) > registered default > LATEST
    fn resolve(&self, domain: &str, explicit: Option<u64>) -> u64 {
        explicit
            .or_else(|| THREAD_LOCAL_OPSET.with(|o| o.borrow().get(domain).copied()))
            .or_else(|| self.versions.get(domain).copied())
            .unwrap_or(LATEST_ONNX_OPSET)
    }
}

thread_local! {
    static THREAD_LOCAL_OPSET: RefCell<HashMap<String, u64>> = RefCell::new(HashMap::new());
}

const LATEST_ONNX_OPSET: u64 = 26;
```

---

## 6. Custom Domains

### 6.1 Registration

```python
# Register a custom domain with default opset
nxrt.register_domain("com.microsoft", default_opset=1)
nxrt.register_domain("ai.onnx.ml", default_opset=3)

# Query registered domains
nxrt.domains()
# {'': 26, 'com.microsoft': 1, 'ai.onnx.ml': 3}
```

### 6.2 Domain Namespace

```python
# Create a domain handle for cleaner dispatch
ms = nxrt.domain("com.microsoft")

# dispatch through domain handle (domain param auto-filled)
out = ms.dispatch("FusedMatMul", [a, b], alpha=0.5)

# Can also register convenience ops on domain handle
out = ms.ops.fused_matmul(a, b, alpha=0.5)  # if registered
```

### 6.3 Opset Control Per Domain

```python
# Context manager supports per-domain opset
with nxrt.opset(18, microsoft=2):
    out1 = nxrt.ops.relu(x)                                       # standard opset 18
    out2 = nxrt.dispatch("FusedMatMul", [a, b],
                         domain="com.microsoft", alpha=0.5)        # microsoft opset 2

# Or set domain default
nxrt.register_domain("com.microsoft", default_opset=2)
```

### 6.4 Rust: Domain Registry

```rust
pub struct DomainRegistry {
    /// domain → DomainInfo
    domains: HashMap<String, DomainInfo>,
}

pub struct DomainInfo {
    pub name: String,
    pub default_opset: u64,
    /// Kernel factories registered for this domain.
    pub kernels: KernelRegistry,
}

impl DomainRegistry {
    pub fn register(&mut self, domain: &str, default_opset: u64) {
        self.domains.insert(domain.to_string(), DomainInfo {
            name: domain.to_string(),
            default_opset,
            kernels: KernelRegistry::new(),
        });
    }

    pub fn resolve_opset(&self, domain: &str) -> u64 {
        self.domains.get(domain)
            .map(|d| d.default_opset)
            .unwrap_or(LATEST_ONNX_OPSET)
    }
}
```

---

## 7. Subgraph Ops

### 7.1 Problem

ONNX ops with subgraph attributes: `If`, `Loop`, `Scan`, `SequenceMap`.
In eager mode, subgraphs can't be pre-compiled ONNX graphs — the whole point of
eager is Python-native execution.

### 7.2 Solution: Python Callables as Subgraphs

```python
# If — condition selects a branch
out = nxrt.ops.if_(
    condition,
    then_branch=lambda: nxrt.ops.relu(x),
    else_branch=lambda: nxrt.ops.sigmoid(x),
)

# Loop — iterative execution
def loop_body(i, cond, carry):
    """
    Args:
        i: iteration index (int64 scalar)
        cond: continue condition (bool scalar)
        carry: loop-carried state tensor(s)
    Returns:
        (keep_going, new_carry)
    """
    new_carry = nxrt.ops.add(carry, x)
    keep_going = nxrt.ops.less(i, max_iter)
    return keep_going, new_carry

final_carry = nxrt.ops.loop(
    max_iter=10,
    initial_cond=True,
    carries=[init_state],
    body=loop_body,
)

# Scan — sequential processing along a dimension
def scan_body(state, elem):
    new_state = nxrt.ops.add(state, elem)
    output = new_state
    return new_state, output

final_state, scan_outputs = nxrt.ops.scan(
    initial_state=[init],
    inputs=[sequence],
    body=scan_body,
)
```

### 7.3 `dispatch()` Does Not Support Subgraph Ops

`dispatch()` is for single-kernel ops. Subgraph ops go through `ops.*` only:

```python
# This is an error:
nxrt.dispatch("If", inputs=[cond], then_branch=..., else_branch=...)
# → Error: 'If' has subgraph attributes; use nxrt.ops.if_() instead
```

### 7.4 Callable Contract

```python
# Subgraph callables receive and return PyTensor(s).
# They execute eagerly — each op inside calls dispatch() recursively.

# then_branch / else_branch: () → PyTensor | tuple[PyTensor, ...]
# loop body: (i, cond, *carries) → (keep_going, *new_carries)
# scan body: (*states, *inputs) → (*new_states, *outputs)
```

### 7.5 Rust: SubgraphCallable

```rust
pub enum SubgraphCallable {
    /// Python callable — calls back into Python each iteration.
    /// Flexible but holds GIL.
    Python(PyObject),
    /// Pre-compiled graph — pure Rust execution, no GIL.
    /// For performance-critical paths; user builds via graph API.
    Compiled(CompiledGraph),
}

impl SubgraphCallable {
    fn call(
        &self,
        ctx: &mut EagerContext,
        inputs: &[&Tensor],
    ) -> Result<Vec<Tensor>> {
        match self {
            Self::Python(py_fn) => {
                Python::with_gil(|py| {
                    let py_inputs: Vec<PyObject> = inputs.iter()
                        .map(|t| PyTensor::from((*t).clone()).into_py(py))
                        .collect();
                    let result = py_fn.call1(py, PyTuple::new(py, &py_inputs))?;
                    parse_py_outputs(py, result)
                })
            }
            Self::Compiled(graph) => {
                ctx.execute_subgraph(graph, inputs)
            }
        }
    }
}
```

### 7.6 Performance Implications

| Mode | GIL | CUDAGraph-compatible | Use case |
|------|-----|---------------------|----------|
| Python callable | Held per iteration | No | Debugging, prototyping, dynamic logic |
| Compiled subgraph | Released | Yes | Performance-critical loops |
| Graph mode (nxrt.load) | Released | Yes | Production inference |

**Recommendation:** Eager + Python callable for development. Graph mode for production.

---

## 8. Kernel Registry & Cache

### 8.1 Versioned Kernel Lookup

Kernels register with a `since_version`. Dispatch finds the highest version ≤ requested opset.
Same approach as ORT's kernel registry.

```rust
pub struct KernelRegistry {
    /// (op_type, domain) → Vec<(since_version, KernelFactory)>
    /// Sorted by since_version descending for fast lookup.
    entries: HashMap<(String, String), Vec<(u64, Box<dyn KernelFactory>)>>,
}

impl KernelRegistry {
    /// Register a kernel implementation for an op at a specific opset version.
    pub fn register(
        &mut self,
        op_type: &str,
        domain: &str,
        since_version: u64,
        factory: Box<dyn KernelFactory>,
    ) {
        let key = (op_type.to_string(), domain.to_string());
        let entries = self.entries.entry(key).or_default();
        entries.push((since_version, factory));
        entries.sort_by(|a, b| b.0.cmp(&a.0)); // descending
    }

    /// Find kernel: highest since_version ≤ requested opset.
    pub fn lookup(
        &self,
        op_type: &str,
        domain: &str,
        opset: u64,
    ) -> Option<&dyn KernelFactory> {
        let key = (op_type.to_string(), domain.to_string());
        self.entries.get(&key)?
            .iter()
            .find(|(since, _)| *since <= opset)
            .map(|(_, factory)| factory.as_ref())
    }
}
```

### 8.2 Kernel Cache (Compiled Kernel Reuse)

```rust
pub struct KernelCache {
    /// Compiled kernel cache: (op_key, shapes, device) → ready kernel.
    cache: LruCache<KernelCacheKey, Arc<dyn Kernel>>,
}

#[derive(Hash, Eq, PartialEq)]
pub struct KernelCacheKey {
    pub op_type: String,
    pub domain: String,
    pub opset: u64,
    pub input_shapes: Vec<Vec<usize>>,
    pub input_dtypes: Vec<DataType>,
    pub device: DeviceId,
}

impl KernelCache {
    pub fn new(capacity: usize) -> Self {
        Self { cache: LruCache::new(capacity) }
    }

    pub fn get_or_create(
        &mut self,
        key: KernelCacheKey,
        create: impl FnOnce() -> Result<Arc<dyn Kernel>>,
    ) -> Result<Arc<dyn Kernel>> {
        if let Some(kernel) = self.cache.get(&key) {
            return Ok(kernel.clone());
        }
        let kernel = create()?;
        self.cache.put(key, kernel.clone());
        Ok(kernel)
    }
}
```

---

## 9. Shape Inference

Eager mode doesn't have a graph to run ONNX shape inference on. We need per-op
shape/dtype inference to allocate output tensors before kernel execution.

### 9.1 Versioned Shape Inference

```rust
/// Infer output shapes and dtypes for a single op.
///
/// Must handle opset-version-dependent behavior (e.g. Reshape v5 vs v14).
pub fn infer_output_meta(
    op_type: &str,
    domain: &str,
    opset: u64,
    inputs: &[&TensorMeta],
    attrs: &HashMap<String, Attribute>,
) -> Result<Vec<TensorMeta>> {
    match (domain, op_type) {
        ("", "Reshape") if opset >= 14 => infer_reshape_v14(inputs, attrs),
        ("", "Reshape") => infer_reshape_v5(inputs, attrs),
        ("", "Resize") if opset >= 11 => infer_resize_v11(inputs, attrs),
        ("", "Resize") => infer_resize_v10(inputs, attrs),
        // ... per-op, per-version
        _ => Err(Error::NoShapeInference {
            op_type: op_type.into(),
            domain: domain.into(),
        }),
    }
}

pub struct TensorMeta {
    pub shape: Vec<usize>,
    pub dtype: DataType,
}
```

### 9.2 Fallback: Kernel-Provided Inference

If built-in shape inference doesn't cover an op (especially custom domain ops),
fall back to the kernel itself:

```rust
pub trait KernelFactory: Send + Sync {
    /// Create a kernel instance.
    fn create(&self, attrs: &HashMap<String, Attribute>) -> Result<Box<dyn Kernel>>;

    /// Optional: kernel-provided output shape inference.
    fn infer_outputs(
        &self,
        inputs: &[&TensorMeta],
        attrs: &HashMap<String, Attribute>,
    ) -> Result<Vec<TensorMeta>> {
        Err(Error::NotImplemented)
    }
}
```

---

## 10. Rust Implementation

### 10.1 EagerContext

```rust
/// Global eager execution context.
///
/// Manages EP registry, kernel cache, device detection, and opset resolution.
/// One per process, thread-safe via internal locking.
pub struct EagerContext {
    /// Available EPs, auto-detected at initialization.
    eps: Vec<Arc<dyn ExecutionProvider>>,
    /// Kernel cache (compiled kernels, keyed by op+shape+device).
    cache: Mutex<KernelCache>,
    /// Domain registry (standard + custom).
    domains: RwLock<DomainRegistry>,
    /// Default device for new tensors.
    default_device: DeviceId,
}

impl EagerContext {
    /// Initialize with auto-detected devices.
    pub fn new() -> Result<Self> {
        let eps = detect_available_eps()?;
        let default_device = eps.first()
            .map(|ep| ep.device_id())
            .unwrap_or(DeviceId::cpu());

        let mut domains = DomainRegistry::new();
        domains.register("", LATEST_ONNX_OPSET);        // standard ONNX (26)
        domains.register("ai.onnx.ml", 3);               // ML domain
        domains.register("ai.onnx.training", 1);          // training domain

        Ok(Self {
            eps,
            cache: Mutex::new(KernelCache::new(4096)),
            domains: RwLock::new(domains),
            default_device,
        })
    }

    /// Dispatch a single ONNX op.
    pub fn dispatch(
        &self,
        op_type: &str,
        domain: &str,
        inputs: &[&Tensor],
        attrs: &HashMap<String, Attribute>,
        explicit_opset: Option<u64>,
    ) -> Result<Vec<Tensor>> {
        // 1. Resolve opset version
        let opset = self.domains.read().unwrap()
            .resolve_opset_with_context(domain, explicit_opset);

        // 2. Resolve target device from inputs
        let device = self.resolve_device(inputs)?;

        // 3. Build cache key
        let cache_key = KernelCacheKey {
            op_type: op_type.into(),
            domain: domain.into(),
            opset,
            input_shapes: inputs.iter().map(|t| t.shape().to_vec()).collect(),
            input_dtypes: inputs.iter().map(|t| t.dtype()).collect(),
            device,
        };

        // 4. Get or compile kernel
        let kernel = self.cache.lock().unwrap().get_or_create(cache_key, || {
            let ep = self.ep_for_device(device)?;
            let registry = ep.kernel_registry();
            let factory = registry.lookup(op_type, domain, opset)
                .ok_or_else(|| Error::NoKernel {
                    op_type: op_type.into(),
                    domain: domain.into(),
                    device,
                })?;
            Ok(Arc::from(factory.create(attrs)?))
        })?;

        // 5. Infer output shapes and dtypes
        let input_meta: Vec<TensorMeta> = inputs.iter()
            .map(|t| TensorMeta { shape: t.shape().to_vec(), dtype: t.dtype() })
            .collect();
        let output_meta = infer_output_meta(op_type, domain, opset, &input_meta, attrs)?;

        // 6. Allocate output tensors
        let mut outputs: Vec<Tensor> = output_meta.iter()
            .map(|m| Tensor::zeros(&m.shape, m.dtype, device))
            .collect();

        // 7. Execute
        let input_views: Vec<TensorView> = inputs.iter().map(|t| t.view()).collect();
        let mut output_views: Vec<TensorMut> = outputs.iter_mut().map(|t| t.view_mut()).collect();
        kernel.execute(&input_views, &mut output_views)?;

        Ok(outputs)
    }

    /// Resolve device from inputs. Error on mixed devices.
    fn resolve_device(&self, inputs: &[&Tensor]) -> Result<DeviceId> {
        let devices: HashSet<DeviceId> = inputs.iter().map(|t| t.device()).collect();
        match devices.len() {
            0 => Ok(self.default_device),
            1 => Ok(*devices.iter().next().unwrap()),
            _ => Err(Error::MixedDeviceInputs {
                devices: devices.into_iter().collect(),
                hint: "Use .to(device) to move tensors to the same device".into(),
            }),
        }
    }

    /// Get EP for a device.
    fn ep_for_device(&self, device: DeviceId) -> Result<&dyn ExecutionProvider> {
        self.eps.iter()
            .find(|ep| ep.device_id() == device)
            .map(|ep| ep.as_ref())
            .ok_or(Error::NoEpForDevice(device))
    }
}
```

### 10.2 Global Singleton

```rust
use std::sync::OnceLock;

static GLOBAL_CONTEXT: OnceLock<EagerContext> = OnceLock::new();

/// Get or initialize the global eager context.
pub fn global_context() -> &'static EagerContext {
    GLOBAL_CONTEXT.get_or_init(|| {
        EagerContext::new().expect("Failed to initialize eager context")
    })
}
```

---

## 11. PyO3 Binding Layer

### 11.1 Module Structure

```rust
#[pymodule]
fn nxrt(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core types
    m.add_class::<PyTensor>()?;

    // Tensor creation
    m.add_function(wrap_pyfunction!(tensor, m)?)?;
    m.add_function(wrap_pyfunction!(from_dlpack, m)?)?;
    m.add_function(wrap_pyfunction!(zeros, m)?)?;
    m.add_function(wrap_pyfunction!(ones, m)?)?;

    // Dispatch
    m.add_function(wrap_pyfunction!(dispatch, m)?)?;

    // Opset control
    m.add_function(wrap_pyfunction!(register_domain, m)?)?;
    m.add_function(wrap_pyfunction!(domains, m)?)?;
    m.add_class::<PyOpsetContext>()?;  // context manager

    // Device context
    m.add_class::<PyDeviceContext>()?;

    // nxrt.ops submodule
    let ops = PyModule::new(m.py(), "ops")?;
    register_ops(&ops)?;
    m.add_submodule(&ops)?;

    // Data types as module-level constants
    m.add("float32", DataTypeWrapper(DataType::Float32))?;
    m.add("float16", DataTypeWrapper(DataType::Float16))?;
    m.add("bfloat16", DataTypeWrapper(DataType::BFloat16))?;
    m.add("int64", DataTypeWrapper(DataType::Int64))?;
    m.add("int32", DataTypeWrapper(DataType::Int32))?;
    m.add("int8", DataTypeWrapper(DataType::Int8))?;
    m.add("bool", DataTypeWrapper(DataType::Bool))?;
    // ...

    Ok(())
}
```

### 11.2 `dispatch()` Binding

```rust
/// nxrt.dispatch(op_type, inputs, /, *, domain="", opset=None, **attrs)
#[pyfunction]
#[pyo3(signature = (op_type, inputs, /, *, domain="", opset=None, **kwargs))]
fn dispatch(
    py: Python<'_>,
    op_type: &str,
    inputs: Vec<PyRef<'_, PyTensor>>,
    domain: &str,
    opset: Option<u64>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyObject> {
    let ctx = global_context();

    // Parse remaining kwargs as ONNX attributes
    let attrs = parse_attributes(kwargs)?;

    // Convert inputs
    let input_refs: Vec<&Tensor> = inputs.iter().map(|t| t.inner()).collect();

    // Dispatch
    let outputs = ctx.dispatch(op_type, domain, &input_refs, &attrs, opset)
        .map_err(to_py_err)?;

    // Single output → tensor, multiple → tuple
    match outputs.len() {
        1 => Ok(PyTensor::from(outputs.into_iter().next().unwrap()).into_pyobject(py)?.into()),
        _ => {
            let py_tensors: Vec<PyObject> = outputs.into_iter()
                .map(|t| PyTensor::from(t).into_pyobject(py).unwrap().into())
                .collect();
            Ok(PyTuple::new(py, &py_tensors)?.into())
        }
    }
}
```

### 11.3 `ops.*` Registration (Macro-Driven)

```rust
/// Macro to generate ops.* wrapper functions with typed signatures.
macro_rules! register_unary_op {
    ($ops:expr, $name:ident, $onnx_name:literal) => {
        #[pyfunction]
        fn $name(x: &PyTensor) -> PyResult<PyTensor> {
            let ctx = global_context();
            let outputs = ctx.dispatch($onnx_name, "", &[x.inner()], &HashMap::new(), None)
                .map_err(to_py_err)?;
            Ok(PyTensor::from(outputs.into_iter().next().unwrap()))
        }
        $ops.add_function(wrap_pyfunction!($name, $ops)?)?;
    };
}

macro_rules! register_binary_op {
    ($ops:expr, $name:ident, $onnx_name:literal) => {
        #[pyfunction]
        fn $name(a: &PyTensor, b: &PyTensor) -> PyResult<PyTensor> {
            let ctx = global_context();
            let outputs = ctx.dispatch($onnx_name, "", &[a.inner(), b.inner()], &HashMap::new(), None)
                .map_err(to_py_err)?;
            Ok(PyTensor::from(outputs.into_iter().next().unwrap()))
        }
        $ops.add_function(wrap_pyfunction!($name, $ops)?)?;
    };
}

fn register_ops(ops: &Bound<'_, PyModule>) -> PyResult<()> {
    // Unary
    register_unary_op!(ops, relu, "Relu");
    register_unary_op!(ops, sigmoid, "Sigmoid");
    register_unary_op!(ops, tanh, "Tanh");

    // Binary
    register_binary_op!(ops, add, "Add");
    register_binary_op!(ops, sub, "Sub");
    register_binary_op!(ops, mul, "Mul");
    register_binary_op!(ops, div, "Div");
    register_binary_op!(ops, matmul, "MatMul");

    // Ops with parameters need manual registration
    // (softmax, layer_norm, reshape, etc.)
    ops.add_function(wrap_pyfunction!(softmax, ops)?)?;
    ops.add_function(wrap_pyfunction!(layer_norm, ops)?)?;
    ops.add_function(wrap_pyfunction!(transpose, ops)?)?;
    ops.add_function(wrap_pyfunction!(reshape, ops)?)?;
    // ...

    Ok(())
}
```

---

## 12. Crate & Directory Structure

```
onnx-genai/
├── crates/
│   ├── onnx-runtime-eager/              # Eager execution engine
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs                   # EagerContext, global singleton
│   │       ├── dispatch.rs              # Core dispatch logic
│   │       ├── shape_inference.rs       # Per-op output shape/dtype inference
│   │       ├── shape_inference/
│   │       │   ├── mod.rs
│   │       │   ├── math.rs              # MatMul, Gemm, element-wise
│   │       │   ├── nn.rs                # Conv, BatchNorm, LayerNorm
│   │       │   ├── manipulation.rs      # Reshape, Transpose, Concat
│   │       │   └── reduction.rs         # ReduceSum, ReduceMean, etc.
│   │       ├── device.rs                # Device resolution, auto-transfer
│   │       ├── opset.rs                 # Opset context, versioning logic
│   │       ├── domain.rs                # Domain registry
│   │       ├── subgraph.rs              # If/Loop/Scan eager execution
│   │       └── cache.rs                 # KernelCache (LRU)
│   ├── onnx-runtime-ir/                 # Graph IR (from ORT2.md §3)
│   ├── onnx-runtime-ep-api/             # EP trait + kernel trait
│   └── ...
├── bindings/
│   └── python/
│       ├── Cargo.toml                   # PyO3 + maturin
│       ├── pyproject.toml               # Python package config
│       └── src/
│           ├── lib.rs                   # nxrt module definition
│           ├── tensor.rs                # PyTensor, DLPack, numpy
│           ├── eager.rs                 # dispatch() binding
│           ├── opset.rs                 # opset context manager
│           ├── device.rs                # device context manager
│           ├── dtypes.rs                # DataType constants
│           └── ops/
│               ├── mod.rs               # ops submodule + macro-driven registration
│               ├── math.rs              # matmul, add, mul, gemm, ...
│               ├── nn.rs                # relu, softmax, layer_norm, ...
│               ├── manipulation.rs      # transpose, reshape, concat, split, ...
│               └── control_flow.rs      # if_, loop, scan (subgraph ops)
└── ...
```

---

## 13. Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Global context vs explicit | **Global singleton** (thread-safe) | Python users expect implicit context (like PyTorch) |
| Mixed device inputs | **Error** | Silent transfers are performance traps; `.to()` is explicit |
| ops.* signature stability | **Pinned to nxrt major version** | Never surprise users with ONNX opset breaking changes |
| Opset control granularity | **3 levels** (global / context / per-call) | Covers all use cases without complexity |
| Custom domain support | **First-class** (register + namespace + opset) | com.microsoft ops are essential for real workloads |
| Subgraph ops in eager | **Python callables only** | Eager = Python control flow; ONNX subgraph = graph mode |
| dispatch() + subgraph | **Error** | Subgraphs are too complex for generic dispatch |
| Kernel lookup | **since_version ≤ opset** (descending) | Matches ORT; one kernel covers a range of opsets |
| Shape inference | **Built-in + kernel fallback** | Must infer output shapes before allocation |
| Return convention | **Single → Tensor, multi → tuple** | Pythonic; matches PyTorch operator returns |
| Kernel cache | **Per-process LRU (4096)** | Same op+shape reuse across calls; bounded memory |
| ops.* registration | **Macro-driven** | Reduce boilerplate for 100+ ops |
