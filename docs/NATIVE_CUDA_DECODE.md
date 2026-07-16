# Native CUDA Decode Design
**Status:** design for review; this document implements no runtime wiring<br>
**Scope:** pure-Rust native runtime, single-GPU decoder-only inference<br>
**Primary model:** Qwen2.5-0.5B int4 (`qwen2.5-0.5b-int4-onnx`)<br>
**Date:** 2026-07-16

## 1. Executive summary

The native runtime already has a substantial CUDA execution provider, but the
session executor cannot use it. `SessionBuilder` accepts a GPU preference and
then discards it, `InferenceSession::from_parts` always creates the CPU EP, and
`Executor` stores `Arc<CpuExecutionProvider>` rather than the execution-provider
trait ([session builder, lines 451-477](../crates/onnx-runtime-session/src/lib.rs#L451-L477),
[session construction, lines 669-695](../crates/onnx-runtime-session/src/lib.rs#L669-L695),
[executor, lines 209-227](../crates/onnx-runtime-session/src/executor.rs#L209-L227)).

The recommended wiring seam is **dynamic dispatch through one selected
`Arc<dyn ExecutionProvider>` per executor**. The trait is already object-safe,
the tensor owner already stores `Arc<dyn ExecutionProvider>`, and kernels are
already cached as `Box<dyn Kernel>` ([EP trait, lines
258-339](../crates/onnx-runtime-ep-api/src/provider.rs#L258-L339), [tensor owner,
lines 118-141](../crates/onnx-runtime-session/src/tensor.rs#L118-L141), [kernel
cache, lines 145-205](../crates/onnx-runtime-session/src/executor.rs#L145-L205)).
This avoids making the recursive control-flow executor generic, does not couple
the session crate to a closed backend enum, and adds no virtual dispatch inside
CUDA kernels. The cached `Kernel::execute` call is already virtual; EP calls on
the steady decode path are limited to buffer management and cache misses.

This change is necessary but not sufficient. The executor currently assumes
that every buffer is host-readable:

- initializer mmaps are wrapped as though they belonged to the selected EP;
- `write_host` is used for initializer and input population;
- every kernel view is stamped `DeviceId::cpu()`;
- strided materialization dereferences the backing pointer on the host; and
- graph outputs are read back through `host_bytes`.

Those assumptions are visible in the initializer path ([executor, lines
682-728](../crates/onnx-runtime-session/src/executor.rs#L682-L728)), input binding
([lines 1160-1168](../crates/onnx-runtime-session/src/executor.rs#L1160-L1168)),
view construction ([lines 1384-1401](../crates/onnx-runtime-session/src/executor.rs#L1384-L1401),
[lines 1544-1569](../crates/onnx-runtime-session/src/executor.rs#L1544-L1569)),
and output collection ([lines
1210-1241](../crates/onnx-runtime-session/src/executor.rs#L1210-L1241)).
`write_host` and `host_bytes` deliberately reject non-host devices
([tensor.rs, lines 59-109](../crates/onnx-runtime-session/src/tensor.rs#L59-L109)).

The first target should therefore be an **all-CUDA, hard-fail decode plan** for
Qwen2.5-0.5B int4:

1. upload packed int4 weights once at session construction;
2. keep activation buffers and KV cache on CUDA;
3. upload only per-step token/position/sequence metadata;
4. download only logits;
5. fail plan construction if any target node lacks a compatible CUDA kernel;
6. then make the fixed-shape one-token step capturable and replayable.

Do not begin with transparent per-op CPU fallback. Correct fallback requires
multi-EP placement, per-value device ownership, explicit transfer edges, and
capture segmentation. An accidental fallback inside a decoder layer would copy
large activations across PCIe and defeat the purpose of native CUDA decode.

## 2. Goal and scope

### 2.1 Goal

Run the complete native incremental decode step for a decoder LLM on one NVIDIA
GPU, beginning with Qwen2.5-0.5B int4:

```text
CPU token selection
  -> H2D: next input id + position/sequence metadata
  -> CUDA: embedding, 24 decoder layers, LM head
  -> device-resident KV append
  -> D2H: logits
  -> CPU sampling
```

The success criterion is not merely “a CUDA kernel was called.” The steady-state
decode loop must keep weights, activations, and KV on device, with transfer
volume independent of context length. CUDA graph replay is the final launch-
overhead reduction once the uncaptured device-resident path is correct.

### 2.2 In scope

- one CUDA device and one real, non-default CUDA stream per native session;
  before graph capture, create it with `CudaContext::new_stream()` and bind the
  cuDNN/cuBLAS handles to it rather than using the legacy/null default stream;
- batch size 1 first, fixed single-token decode shape;
- prompt/prefill may initially use the uncaptured CUDA path;
- Qwen2.5-0.5B int4, f32 activations/KV, packed QKV GQA;
- synchronous H2D/D2H for correctness in M1/M2, then stream-ordered transfers;
- hard-fail CUDA placement for the target decode graph;
- device-resident static-capacity KV with a logical length cursor;
- CUDA graph capture/replay of the one-token decode step.

### 2.3 Out of scope

- multi-GPU tensor, pipeline, expert, or sequence parallelism;
- training, gradients, optimizer state, or checkpointing;
- paged attention and continuous batching in the first implementation;
- speculative decoding integration;
- GPU sampling/tokenization;
- general heterogeneous placement for arbitrary ONNX graphs;
- control-flow and sequence operators on CUDA.

The design should leave room for those features, but none should delay the
single-GPU decoder path.

## 3. Audited current state

### 3.1 Target graph

Inspection of
`/home/justinchu/qwen2.5-0.5b-int4-onnx/model.onnx` on 2026-07-16 found 299
nodes and these 13 `(domain, op_type)` pairs:

| Domain / op | Nodes | Decode role |
|---|---:|---|
| `com.microsoft::MatMulNBits` | 121 | packed-int4 projections and LM head |
| `com.microsoft::SkipSimplifiedLayerNormalization` | 48 | fused residual + RMSNorm |
| `ai.onnx::Mul` | 48 | SwiGLU and mask arithmetic |
| `ai.onnx::Add` | 24 | packed-QKV bias |
| `ai.onnx::Sigmoid` | 24 | decomposed SiLU |
| `com.microsoft::GroupQueryAttention` | 24 | packed QKV, RoPE, attention, KV |
| `ai.onnx::Cast` | 2 | mask metadata conversion |
| `ai.onnx::Constant` | 2 | mask-subgraph constants |
| `ai.onnx::Gather` | 2 | token embedding and shape indexing |
| `ai.onnx::ReduceSum` | 1 | valid sequence length from mask |
| `ai.onnx::Shape` | 1 | attention-mask capacity |
| `ai.onnx::SimplifiedLayerNormalization` | 1 | initial RMSNorm |
| `ai.onnx::Sub` | 1 | mask length adjustment |

The earlier gap report records the same graph size and op distribution before
CUDA `Gather`, `Shape`, and `Constant` landed ([CUDA gap report, lines
38-68](benchmarks/2026-07-16-cuda-int4-decode.md#L38-L68)). Current source now
registers all three ([CUDA registry, lines
161-186](../crates/onnx-runtime-ep-cuda/src/kernels/mod.rs#L161-L186)).

The representative GQA node uses packed QKV: input 0 is present while key and
value inputs 1 and 2 are omitted. It has `num_heads=14`, `kv_num_heads=2`,
`do_rotary=1`, and receives cosine/sine caches. This matters because the current
CUDA GQA kernel explicitly rejects omitted key/value inputs and requires
unpacked rank-3 Q, K, and V ([group_query_attention.rs, lines
290-332](../crates/onnx-runtime-ep-cuda/src/kernels/group_query_attention.rs#L290-L332)).
Therefore “GQA is registered” does **not** yet mean the target model is
executable.

### 3.2 CUDA EP inventory

The CUDA crate already contains the principal backend layers:

| Source | What exists today |
|---|---|
| `provider.rs` | `CudaExecutionProvider`, registry-based kernel matching, CUDA allocation/free, D2D copy, and stream synchronization ([lines 57-85](../crates/onnx-runtime-ep-cuda/src/provider.rs#L57-L85), [lines 87-157](../crates/onnx-runtime-ep-cuda/src/provider.rs#L87-L157), [lines 159-229](../crates/onnx-runtime-ep-cuda/src/provider.rs#L159-L229)). |
| `runtime.rs` | one context/default stream, cuBLASLt, cuDNN, NVRTC module cache, raw allocation, H2D, D2H, and D2D helpers ([lines 49-88](../crates/onnx-runtime-ep-cuda/src/runtime.rs#L49-L88), [lines 165-233](../crates/onnx-runtime-ep-cuda/src/runtime.rs#L165-L233)). |
| `blas.rs` | full-f32-accumulating cuBLASLt GEMM plumbing, including row-major mapping and fused epilogues ([lines 1-47](../crates/onnx-runtime-ep-cuda/src/blas.rs#L1-L47), [lines 59-100](../crates/onnx-runtime-ep-cuda/src/blas.rs#L59-L100)). |
| `cudnn/` | lazily created, stream-bound cuDNN backend for softmax, reduction, convolution, and pooling ([cudnn/mod.rs, lines 1-21](../crates/onnx-runtime-ep-cuda/src/cudnn/mod.rs#L1-L21), [lines 343-471](../crates/onnx-runtime-ep-cuda/src/cudnn/mod.rs#L343-L471), [lines 618-661](../crates/onnx-runtime-ep-cuda/src/cudnn/mod.rs#L618-L661)). |
| `capture.rs` | an all-kernels eligibility predicate; it does **not** yet begin, instantiate, launch, or destroy CUDA graphs ([lines 1-11](../crates/onnx-runtime-ep-cuda/src/capture.rs#L1-L11)). |
| `kernels/` | 65 advertised op names, including transformer-critical matmul, int4 matmul, GQA, normalization, elementwise, cast, gather, shape, constant, reduction, and softmax ([mod.rs, lines 49-145](../crates/onnx-runtime-ep-cuda/src/kernels/mod.rs#L49-L145)). |

`CudaExecutionProvider::copy` is currently D2D-only in practice: it asserts that
both buffers belong to the CUDA device ([provider.rs, lines
192-218](../crates/onnx-runtime-ep-cuda/src/provider.rs#L192-L218)). The trait
describes `copy` as host/device-capable, but its two `DeviceBuffer` arguments do
not identify two owning EPs ([EP API, lines
283-294](../crates/onnx-runtime-ep-api/src/provider.rs#L283-L294)). The CUDA
runtime has concrete H2D/D2H helpers, but those helpers are unavailable through
`dyn ExecutionProvider`. A device-polymorphic executor therefore needs an
explicit host-transfer API before CUDA can be wired safely.

### 3.3 Target op readiness

“Kernel exists” and “ready for end-to-end decode” are different states:

| Target operation | CUDA implementation | Target readiness | Capture readiness |
|---|---|---|---|
| `MatMulNBits` | Registered. Supports standard 4-bit layout, f32 activation/output, optional zp/g_idx/bias ([matmul_nbits.rs, lines 59-96](../crates/onnx-runtime-ep-cuda/src/kernels/matmul_nbits.rs#L59-L96), [lines 108-203](../crates/onnx-runtime-ep-cuda/src/kernels/matmul_nbits.rs#L108-L203)). | **Correctness candidate, not decode-efficient.** It allocates a full `K*N` f32 weight and cuBLAS workspace, dequantizes, runs GEMM, synchronizes, and frees on every invocation ([lines 205-263](../crates/onnx-runtime-ep-cuda/src/kernels/matmul_nbits.rs#L205-L263)). Packed weights can be uploaded once, but the current compute path still expands them every token. | **No.** Explicitly false ([lines 324-335](../crates/onnx-runtime-ep-cuda/src/kernels/matmul_nbits.rs#L324-L335)). |
| `GroupQueryAttention` | Registered. Includes cache construction, RoPE, attention, and BNSH/BSH transforms ([group_query_attention.rs, lines 19-98](../crates/onnx-runtime-ep-cuda/src/kernels/group_query_attention.rs#L19-L98)). | **Blocked for the target.** Packed QKV is rejected. The current cache kernel also scans and rewrites the complete present cache, rather than appending one token ([lines 33-54](../crates/onnx-runtime-ep-cuda/src/kernels/group_query_attention.rs#L33-L54), [lines 615-645](../crates/onnx-runtime-ep-cuda/src/kernels/group_query_attention.rs#L615-L645)). | **No.** It performs D2H reads of sequence/position metadata and per-call scratch allocations ([lines 236-260](../crates/onnx-runtime-ep-cuda/src/kernels/group_query_attention.rs#L236-L260), [lines 526-551](../crates/onnx-runtime-ep-cuda/src/kernels/group_query_attention.rs#L526-L551)); it does not override the default `cuda_graph_compatible=false` ([EP kernel API, lines 165-168](../crates/onnx-runtime-ep-api/src/kernel.rs#L165-L168)). |
| RMSNorm / `SimplifiedLayerNormalization` | A fused f32 kernel exists, but the registry maps standard-domain `RMSNormalization` and `com.microsoft::SimplifiedLayerNormalization` only ([registry, lines 322-335](../crates/onnx-runtime-ep-cuda/src/kernels/mod.rs#L322-L335), [normalization.rs, lines 479-609](../crates/onnx-runtime-ep-cuda/src/kernels/normalization.rs#L479-L609)). | **Missing target registration.** The inspected model uses standard-domain `ai.onnx::SimplifiedLayerNormalization`; add the exact domain/opset registration after verifying its contract matches the existing kernel. | The kernel advertises true, but its body synchronizes after launch ([lines 578-608](../crates/onnx-runtime-ep-cuda/src/kernels/normalization.rs#L578-L608)); the capture implementation must remove per-op synchronization. |
| `SkipSimplifiedLayerNormalization` | Registered fused residual RMSNorm ([registry, lines 337-349](../crates/onnx-runtime-ep-cuda/src/kernels/mod.rs#L337-L349)). | **Available**; GPU tests already cover its output contract. Verify the target's omitted optional outputs. | Advertised true; audit synchronization before capture. |
| RoPE | No standalone `RotaryEmbedding` registration. | **Available only inside CUDA GQA** when `do_rotary=1`; that is sufficient for this target after packed-QKV support lands ([group_query_attention.rs, lines 457-520](../crates/onnx-runtime-ep-cuda/src/kernels/group_query_attention.rs#L457-L520), [lines 648-671](../crates/onnx-runtime-ep-cuda/src/kernels/group_query_attention.rs#L648-L671)). A future decomposed-RoPE export remains a gap. | Blocked with GQA. |
| SiLU / activation | `Sigmoid` exists; `Mul` exists; no `Silu`/`Swish` is registered in the CUDA registry ([registry, lines 224-278](../crates/onnx-runtime-ep-cuda/src/kernels/mod.rs#L224-L278)). | **Prerequisite.** `Executor::build` always rewrites the target's `x*Sigmoid(x)` pattern to `Silu` before planning ([executor, lines 602-675](../crates/onnx-runtime-session/src/executor.rs#L602-L675)). Either add CUDA `Silu` or make that rewrite conditional on selected-EP support. Adding the trivial fused kernel is preferred. | A pure pointwise SiLU launch should be capturable after warmup and after removing per-op stream synchronization. |
| `Add` / `Mul` / `Sub` | Registered broadcast binary kernel ([elementwise.rs, lines 356-473](../crates/onnx-runtime-ep-cuda/src/kernels/elementwise.rs#L356-L473)). | **Available** for target dtypes/shapes; verify all broadcast forms. | **No today.** It allocates/uploads/frees broadcast metadata every call and returns false ([lines 421-458](../crates/onnx-runtime-ep-cuda/src/kernels/elementwise.rs#L421-L458), [lines 471-473](../crates/onnx-runtime-ep-cuda/src/kernels/elementwise.rs#L471-L473)). Cache metadata in the compiled kernel or use fixed inline parameters. |
| `Cast` | Registered broad dtype conversion ([registry, lines 351-359](../crates/onnx-runtime-ep-cuda/src/kernels/mod.rs#L351-L359)). | **Available**; verify the mask subgraph's exact int64/int32 conversions. | Advertised true, but execution synchronizes ([cast.rs, lines 290-314](../crates/onnx-runtime-ep-cuda/src/kernels/cast.rs#L290-L314)). |
| `Gather` / embedding | Registered axis-parametric device kernel ([gather.rs, lines 61-139](../crates/onnx-runtime-ep-cuda/src/kernels/gather.rs#L61-L139)). | **Available** for int64 token indices and f32 embedding data. | **No.** It downloads indices for bounds validation and synchronizes ([lines 140-167](../crates/onnx-runtime-ep-cuda/src/kernels/gather.rs#L140-L167), [lines 218-230](../crates/onnx-runtime-ep-cuda/src/kernels/gather.rs#L218-L230)). For captured decode, validate token ids at the host boundary and use a trusted no-D2H execution path, or perform device-side validation without host sync. |
| `Softmax` | Registered cuDNN/NVRTC implementation ([registry, lines 294-309](../crates/onnx-runtime-ep-cuda/src/kernels/mod.rs#L294-L309)). | The target has no standalone Softmax node; GQA uses its attention engine. Keep standalone support for other exports. | Advertised true ([softmax.rs, lines 332-343](../crates/onnx-runtime-ep-cuda/src/kernels/softmax.rs#L332-L343)); audit synchronization. |
| `Shape` / `Constant` | Registered host-compute + H2D kernels ([shape.rs, lines 31-80](../crates/onnx-runtime-ep-cuda/src/kernels/shape.rs#L31-L80), [constant.rs, lines 59-93](../crates/onnx-runtime-ep-cuda/src/kernels/constant.rs#L59-L93)). | **Available**, but constant folding should remove `Constant` where possible. | Default false. Prefer load-time constant folding and static metadata specialization before capture. |
| `ReduceSum` | Registered CUDA reduction ([registry, lines 361-385](../crates/onnx-runtime-ep-cuda/src/kernels/mod.rs#L361-L385)). | **Available**; verify the one mask reduction. | Verify. Capture metadata alone is not proof that cuDNN workspace/descriptor behavior is capture-safe. |

The largest correctness prerequisite is **target-compatible packed-QKV GQA**;
the smaller standard-domain `SimplifiedLayerNormalization` registration gap and
the executor-introduced `Silu` gap must also be closed in M2.
The largest performance prerequisite is **a decode-specialized int4 projection
path that does not expand every weight matrix to f32 on every token**.

### 3.4 The executor wiring gap

The executor's module comment claims execution-provider generality, but concrete
types and CPU assumptions remain throughout ([executor, lines
1-11](../crates/onnx-runtime-session/src/executor.rs#L1-L11)):

1. `KernelCache::get_or_create` takes `&CpuExecutionProvider` ([lines
   162-174](../crates/onnx-runtime-session/src/executor.rs#L162-L174)).
2. `Executor.ep` and `Executor::build` use `Arc<CpuExecutionProvider>` ([lines
   211-227](../crates/onnx-runtime-session/src/executor.rs#L211-L227), [lines
   666-672](../crates/onnx-runtime-session/src/executor.rs#L666-L672)).
3. `auto_detect_cpu_ep` is the only construction path ([lines
   3126-3133](../crates/onnx-runtime-session/src/executor.rs#L3126-L3133)).
4. Every `TensorView`/`TensorMut` is marked CPU, even when its pointer would be a
   CUDA device address ([lines
   1392-1400](../crates/onnx-runtime-session/src/executor.rs#L1392-L1400), [lines
   1596-1603](../crates/onnx-runtime-session/src/executor.rs#L1596-L1603)).
5. Initializers may borrow host mmap bytes while labeling the buffer with
   `ep.device_id()` ([lines
   703-723](../crates/onnx-runtime-session/src/executor.rs#L703-L723)). That is
   valid for CPU/unified memory, but a CUDA kernel cannot treat a normal host
   mmap pointer as a CUDA allocation.
6. Dynamic-shape value reads, view materialization, control flow, sequence ops,
   and graph-output collection use host dereferences ([lines
   1525-1541](../crates/onnx-runtime-session/src/executor.rs#L1525-L1541), [lines
   1643-1653](../crates/onnx-runtime-session/src/executor.rs#L1643-L1653), [lines
   2334-2412](../crates/onnx-runtime-session/src/executor.rs#L2334-L2412)).

The M1 skeleton must fix these facts, not merely change the field type.

## 4. Recommended execution-provider seam

### 4.1 Options

#### Option A — `Executor<EP: ExecutionProvider>`

```rust
struct Executor<EP: ExecutionProvider> {
    ep: Arc<EP>,
    // ...
}
```

**Advantages**

- statically dispatched EP methods;
- backend-specific methods could be exposed through bounds;
- no EP vtable call for allocation, support checks, or kernel creation.

**Costs**

- `CompiledSubgraph` recursively owns `Executor`, so the generic parameter
  propagates through control-flow state ([executor, lines
  321-331](../crates/onnx-runtime-session/src/executor.rs#L321-L331));
- `InferenceSession`, C API, Python bindings, tests, and constructors either
  become generic or need a type-erased wrapper;
- every EP monomorphizes the 3,000-line executor, increasing build time and code
  size;
- it still does not solve future per-node heterogeneous placement;
- kernel execution remains dynamic through `Box<dyn Kernel>`, so the hot math
  call is not devirtualized.

This is appropriate for a small backend-specific runtime, not for this session
surface.

#### Option B — `Arc<dyn ExecutionProvider>`

```rust
struct Executor {
    ep: Arc<dyn ExecutionProvider>,
    // ...
}
```

**Advantages**

- the trait is already object-safe: it has no generic methods, associated types,
  or `Self` return values ([provider.rs, lines
  258-339](../crates/onnx-runtime-ep-api/src/provider.rs#L258-L339));
- `Tensor` already uses exactly this ownership form ([tensor.rs, lines
  118-141](../crates/onnx-runtime-session/src/tensor.rs#L118-L141));
- the eager runtime already stores EPs as `Vec<Arc<dyn ExecutionProvider>>`
  ([eager context, lines
  59-89](../crates/onnx-runtime-eager/src/lib.rs#L59-L89));
- small surgical diff through `Executor`, `KernelCache`, child executors, and
  session construction;
- naturally extends later to an EP registry and `NodePlan.ep_id`.

**Costs**

- one vtable dispatch for EP operations;
- backend-specific functionality must be added to the trait or a separate
  capability interface;
- initialization must occur before placing the provider in an `Arc`, because
  `initialize` takes `&mut self`.

**Assumption (validate in M1):** cache hits invoke the already-cached `dyn
Kernel`; they do not repeatedly ask the EP to match the node. The structural
claim is established, but the `Arc<dyn ExecutionProvider>` virtual-call cost has
not yet been benchmarked. Do not assume it is negligible relative to a kernel
launch until M1 measures it; it remains absent from the elementwise/GEMM inner
loops.

#### Option C — closed enum

```rust
enum SessionEp {
    Cpu(Arc<CpuExecutionProvider>),
    Cuda(Arc<CudaExecutionProvider>),
}
```

**Advantages**

- explicit exhaustive dispatch;
- backend-specific access without downcasting;
- no public generic propagation.

**Costs**

- session must depend directly on every backend crate;
- each new EP requires editing the enum and every forwarding method;
- duplicates the existing `ExecutionProvider` abstraction;
- incompatible with plugin/third-party EPs and the existing dynamic EP registry
  ([registry.rs, lines
  94-153](../crates/onnx-runtime-ep-api/src/registry.rs#L94-L153)).

This should be rejected.

### 4.2 Recommendation

Use **Option B: `Arc<dyn ExecutionProvider>`**.

M1 should make these exact changes:

```text
KernelCache::get_or_create(..., ep: &dyn ExecutionProvider)
Executor.ep: Arc<dyn ExecutionProvider>
Executor::build(..., ep: Arc<dyn ExecutionProvider>)
CompiledSubgraph reuses the same Arc<dyn ExecutionProvider>
InferenceSession::from_parts selects and initializes CPU or CUDA before build
```

The first CUDA executor still owns one EP for the whole graph. Do not combine
this mechanical polymorphism step with a general placement planner.

Provider construction should remain feature-gated:

```text
onnx-runtime-session/default  -> CPU EP only
onnx-runtime-session/cuda     -> optional onnx-runtime-ep-cuda dependency
DevicePreference::Gpu         -> CUDA EP or an actionable unavailable error
DevicePreference::Auto        -> CUDA when compiled+initializable, else CPU
```

Today the session crate depends only on the CPU EP
([session Cargo.toml, lines
11-18](../crates/onnx-runtime-session/Cargo.toml#L11-L18)), while the engine and
benchmark already carry optional CUDA EP dependencies ([engine Cargo.toml, lines
11-33](../crates/onnx-genai-engine/Cargo.toml#L11-L33), [benchmark Cargo.toml,
lines 10-34](../crates/onnx-genai-bench/Cargo.toml#L10-L34)). Wire those features
through to `onnx-runtime-session/cuda`; do not make CUDA a default build
dependency.

### 4.3 Required transfer seam

Add object-safe host transfer operations to `ExecutionProvider`, for example:

```rust
fn copy_from_host(
    &self,
    src: &[u8],
    dst: &mut DeviceBuffer,
) -> Result<()>;

fn copy_to_host(
    &self,
    src: &DeviceBuffer,
    dst: &mut [u8],
) -> Result<()>;
```

CPU implementations use normal memory copies. CUDA delegates to
`CudaRuntime::htod`/`dtoh`. M1 may make both synchronous; later variants can
return a real `Fence`. Keep D2D `copy` separate so buffer ownership remains
unambiguous.

This API is preferable to downcasting `dyn ExecutionProvider` to CUDA in the
session crate. It also makes the true graph boundaries auditable.

### 4.4 Device-correct executor rules

After polymorphism:

1. Borrow initializer storage only when the selected device is host-accessible.
   CUDA initializers are allocated and uploaded once.
2. Stamp views from the backing buffer's `DeviceId`, never a literal CPU id.
3. Replace `write_host` at graph inputs with `ep.copy_from_host`.
4. Replace output `host_bytes` with `ep.copy_to_host`, then construct a
   CPU-owned result tensor for the current native engine.
5. Reject host-only strided materialization on CUDA until a device materializer
   exists.
6. Reject control-flow/sequence plans in the first CUDA mode.
7. Audit every dynamic-shape read. Small shape tensors either remain in a
   host-side shape plan or are explicitly downloaded; never dereference a CUDA
   pointer.
8. Keep buffer addresses stable across same-shape runs, as the existing
   `ensure_buffer` reuse rule already intends ([executor, lines
   863-879](../crates/onnx-runtime-session/src/executor.rs#L863-L879)).

## 5. Device memory and KV cache

### 5.1 Weight residency

All immutable initializers must be copied to CUDA once during session build and
kept until executor teardown:

```text
external mmap / inline bytes (host)
  -> one H2D upload at build
  -> packed int4/scales/bias/cos/sin device buffers
  -> reused by every prefill/decode run
```

Do not borrow the host mmap as a CUDA `DeviceBuffer`. Do not re-upload weights
per run. Record uploaded bytes and upload count in debug statistics so tests can
prove one-time residency.

For `MatMulNBits`, “weight residency” initially means the packed uint8 weights
and f32 scales. Caching a full f32 dequantization would expand int4 payloads by
roughly 8× before scales and is not the strategic solution. M2 may use the
current dequantize+cuBLAS path to establish correctness; M5 should replace the
decode `M=1` path with a direct packed-int4 GEMV/GEMM kernel or CUTLASS-style
weight-only kernel.

### 5.2 Activation buffers

The executor's value-buffer map can remain the first activation allocator:

- static/fixed decode shapes allocate once;
- identical shapes reuse the same allocation;
- every interior value remains on the selected CUDA device;
- no tensor is materialized to host between nodes.

M5 should add liveness-based reuse or a CUDA arena. CUDA graph capture requires
that all captured addresses remain stable, so arena compaction/replanning must
happen before capture, never between replays.

### 5.3 Current native KV behavior is not acceptable

`NativeDecodeSession` currently owns past tensors in a host `HashMap`, creates a
full attention mask on the host, passes all past tensors through
`InferenceSession::run`, collects every present output, and rotates those
outputs into the next step ([native_decode.rs, lines
225-307](../crates/onnx-genai-engine/src/native_decode.rs#L225-L307)). On CPU
this is correct. On CUDA, if implemented through ordinary graph boundaries, it
would cause context-sized KV transfers every token.

The CUDA GQA kernel also currently builds each present cache by scanning the
entire present capacity and copying old entries before inserting current K/V
([group_query_attention.rs, lines
33-54](../crates/onnx-runtime-ep-cuda/src/kernels/group_query_attention.rs#L33-L54)).
That remains O(context) or O(capacity) device work even if no PCIe transfer
occurs.

### 5.4 Required device-KV design

Introduce native decode state owned alongside the session:

```text
DecodeCudaState
  logical_len: usize
  max_len: usize
  per-layer key:   DeviceBuffer [B, Hkv, max_len, D]
  per-layer value: DeviceBuffer [B, Hkv, max_len, D]
```

For each one-token step:

1. compute K/V for the new token;
2. write them directly to slice `logical_len`;
3. run attention over prefix `0..=logical_len`;
4. increment `logical_len`;
5. retain the same buffer addresses.

The graph-facing past/present values must alias these buffers. That requires a
session binding mechanism capable of:

- supplying an externally owned device buffer for a graph input;
- binding a graph output to the same allocation;
- distinguishing physical capacity from logical valid length;
- suppressing graph-output materialization for bound KV values.

The first implementation should use one max-capacity allocation per key/value
tensor and a scalar logical cursor. Rewind changes the cursor; reset sets it to
zero. Neither operation copies KV bytes. Clearing bytes is unnecessary if every
attention read is bounded by the logical length.

For the target graph, a fixed-capacity attention mask makes physical capacity
and valid sequence length different. The current CUDA GQA implementation
requires `total_sequence_length == max(seqlens_k + 1)` and derives present
capacity from that value ([group_query_attention.rs, lines
397-443](../crates/onnx-runtime-ep-cuda/src/kernels/group_query_attention.rs#L397-L443)).
M3 must split these concepts: the Shape-derived scalar may describe buffer
capacity, while the ReduceSum-derived value describes the valid prefix.

For this Qwen geometry, f32 KV costs:

```text
bytes/token = layers * 2(K,V) * kv_heads * head_dim * sizeof(f32)
            = 24 * 2 * 2 * 64 * 4
            = 24,576 bytes/token

4096 tokens  = 96 MiB
32768 tokens = 768 MiB
```

The maximum context therefore needs an explicit product decision; blindly
allocating the model's theoretical maximum is not free.

### 5.5 Per-step transfer boundary

The desired steady-state boundary is:

| Direction | Data | Size behavior |
|---|---|---|
| H2D | input id | O(1), normally one int64 |
| H2D | position id and/or sequence-length scalars | O(1) |
| H2D | attention-mask update, if retained | O(1) delta into a persistent fixed-capacity mask |
| D2H | logits | O(vocabulary), currently `[1,1,151936]` f32 for the target |
| none | weights | resident after build |
| none | activations | interior device buffers |
| none | KV | resident and appended in place |

The current native adapter constructs an attention-mask vector of
`total_len` on every step ([native_decode.rs, lines
235-253](../crates/onnx-genai-engine/src/native_decode.rs#L235-L253)). For graph
capture, replace that with a fixed-capacity mask whose newly valid element is
updated each token, or specialize the mask subgraph to scalar sequence lengths.

## 6. CUDA graph capture for decode

### 6.1 What exists

`capture.rs` provides one useful invariant:

```rust
subgraph_graph_capturable(kernels)
    == kernels.iter().all(|kernel| kernel.cuda_graph_compatible())
```

([capture.rs, lines 1-11](../crates/onnx-runtime-ep-cuda/src/capture.rs#L1-L11)).
Keep this as the mandatory eligibility gate. It is not a capture executor.

M4 must add CUDA runtime ownership for:

- **Prerequisite 1 — before any capture,** create a real non-default stream with
  `CudaContext::new_stream()`; `CudaRuntime::new` currently uses
  `context.default_stream()`, which is cudarc's legacy/null stream and cannot
  be captured. Bind the cuDNN/cuBLAS handles to this session stream.
- begin stream capture;
- end capture and obtain a graph;
- instantiate a graph executable;
- launch replay on the session stream;
- destroy graph/graph-exec handles before their buffers are freed.

#### 6.1.1 Graph ownership and threading

`ExecutionProvider` requires `Send + Sync`, but cudarc's `CudaGraph` explicitly
implements neither because CUDA graph objects require externally serialized
access. `CudaStream`, in contrast, implements `Send + Sync`. Therefore a
captured graph/graph-exec cannot be a naive field on `CudaRuntime` or
`CudaExecutionProvider`.

M4 needs a deliberate serialized-ownership design and safety justification:

- store the graph behind a `Mutex` in a wrapper with manual `unsafe impl
  Send/Sync`, with the invariant that all graph use is serialized on the single
  owning session/stream;
- keep it in a non-shared, per-session decode driver that is never sent across
  threads; or
- confine interior mutability to the decode loop.

Which ownership model to use is a user/implementation decision.

### 6.2 Capture shape

Capture only the steady one-token step:

```text
input_ids       [1,1]                fixed
position_ids    [1,1] or scalar      fixed
attention_mask  [1,max_len]          fixed physical shape
KV              [1,Hkv,max_len,D]    fixed addresses
logits          [1,1,vocab]          fixed address
```

Prefill runs uncaptured because its sequence shape differs. After prefill has
populated static KV buffers, the first decode step warms every NVRTC module,
descriptor, workspace, and kernel cache entry. A later step captures; subsequent
steps replay.

### 6.3 Required kernel audit

No operation inside capture may:

- allocate or free device memory;
- compile NVRTC;
- create mutable library state lazily;
- perform D2H validation;
- synchronize the capturing stream;
- change tensor shape or address.

Several current kernels violate those rules despite having useful correctness
implementations:

- `MatMulNBits`: per-call weight/workspace allocation;
- GQA: D2H metadata plus multiple scratch allocations;
- binary elementwise: per-call broadcast metadata allocation;
- Gather: D2H index validation;
- Shape/Constant: host-generated uploads;
- many kernels marked compatible still call `runtime.synchronize()` after
  launch.

Therefore M4 is a real kernel-lifecycle refactor, not a wrapper around the run
loop. Treat existing compatibility booleans as claims to verify, not proof.

### 6.4 Lessons from the ORT-backed path

The ORT-backed decoder demonstrates four patterns worth mirroring
conceptually, without reusing its implementation:

1. **Process-unique capture identity.** Each decode session claims a unique
   positive graph id; prefill uses a no-capture sentinel ([ORT decode, lines
   18-37](../crates/onnx-genai-ort/src/decode.rs#L18-L37)).
2. **Persistent fixed-address buffers.** Captured input, mask, logits, and shared
   KV values live for the capture's lifetime ([lines
   138-168](../crates/onnx-genai-ort/src/decode.rs#L138-L168)).
3. **Refresh CPU inputs every step.** ORT's captured graph replays against
   stable internal device inputs, so the CPU bindings must be refreshed or the
   first token freezes ([lines
   431-465](../crates/onnx-genai-ort/src/decode.rs#L431-L465)).
4. **Device KV makes the step O(1) in cache movement.** Shared buffers are
   allocated through the device allocator and bound as both past and present
   ([lines 810-858](../crates/onnx-genai-ort/src/decode.rs#L810-L858)).

For native CUDA, assign a process-unique `CaptureId`, own one graph executable
per live generation, and invalidate/recreate it after reset or a structural
rewind. Release the graph before releasing any referenced buffer.

“Refresh CPU inputs” should mean one explicit operation every replay:

```text
write current token/position into stable host staging
  -> H2D into stable device input buffers
  -> launch graph executable
```

Do not assume graph replay observes changed pageable host memory automatically.
Whether the H2D copies are outside the graph or captured memcpy nodes depends on
the final pinned-memory design, but source and destination addresses must remain
stable.

## 7. Op dispatch and fallback

### 7.1 Existing dispatch

CUDA support matching is registry-based:

- `supports_op` returns `Unsupported` when `(domain, op_type)` is not registered;
- `get_kernel` resolves a factory by op/domain/opset;
- the executor's shape-keyed cache performs the support check and compilation on
  cache miss.

See [CUDA provider, lines
113-157](../crates/onnx-runtime-ep-cuda/src/provider.rs#L113-L157) and [executor
cache, lines 162-205](../crates/onnx-runtime-session/src/executor.rs#L162-L205).

M2 should add a build-time CUDA preflight over the optimized target graph and
report every incompatible node, including shape/attribute incompatibility. A
name in `CUDA_COVERED_OPS` is not sufficient; packed-QKV GQA is the current
counterexample.

`CudaExecutionProvider::supports_op` currently checks registry presence for
almost every op and has an extra shape gate only for fused GEMM
([provider.rs, lines
113-146](../crates/onnx-runtime-ep-cuda/src/provider.rs#L113-L146)). It therefore
claims the packed-QKV GQA node even though `execute` later rejects it. M2 must
make `KernelMatch` honest for supported input forms, not rely on a late runtime
failure.

### 7.2 Recommended fallback policy

For `DevicePreference::Gpu` in the native decoder:

- **decode plan:** hard-fail if any node cannot run on CUDA;
- **diagnostic mode:** optionally print the unsupported node list and suggested
  remediation;
- **CPU request/Auto fallback:** preserve the existing CPU session unchanged.

Do not silently run an unsupported decoder op on CPU.

### 7.3 Future heterogeneous fallback

If general fallback is later required, implement it as a planner:

```text
ordered EP registry
  -> choose EP per node using KernelMatch/cost
  -> assign a device to every value
  -> insert explicit transfer/materialization edges
  -> compile per-EP kernels
```

`EpRegistry::candidates_for_op` already exposes ordered candidates
([registry.rs, lines
94-153](../crates/onnx-runtime-ep-api/src/registry.rs#L94-L153)). Extend
`NodePlan` with an `EpId`; do not discover fallback reactively inside
`exec_kernel_node`.

CUDA graph capture then applies only to a contiguous all-CUDA region. Any CPU
fallback splits capture and introduces synchronization, which is another reason
to require full target coverage first.

## 8. Correctness and validation

### 8.1 CPU preservation

M1 must be behavior-neutral for CPU:

- all existing `onnx-runtime-session` tests pass unchanged;
- CPU graph outputs remain byte-identical;
- borrowed aligned initializer mmap behavior remains enabled only for
  host-accessible EPs;
- cache hit/miss and control-flow reuse tests remain unchanged;
- no CUDA dependency is required for a CPU-only build.

### 8.2 Operator validation

For every target CUDA op:

- compare against the CPU EP or an independent scalar reference;
- test exact target shapes, dtypes, attributes, and optional-input layout;
- test nonzero past length;
- test maximum-capacity boundary;
- test reset and rewind cursor behavior;
- assert no non-finite output.

Use tolerances already established by CUDA tests rather than inventing one
global threshold. Existing tests use approximately `2e-4` for
`MatMulNBits`, `1e-3` for GQA, and `1e-5` for skip-RMSNorm
([matmul_nbits GPU test, lines
253-259](../crates/onnx-runtime-ep-cuda/tests/matmul_nbits_gpu.rs#L253-L259),
[GQA GPU test, line
302](../crates/onnx-runtime-ep-cuda/tests/group_query_attention_gpu.rs#L302),
[skip-RMSNorm GPU test, lines
253-260](../crates/onnx-runtime-ep-cuda/tests/skip_simplified_layer_norm_gpu.rs#L253-L260)).
Full-model logits should report max absolute/relative error and top-k agreement;
greedy token identity is the final behavioral gate.

### 8.3 End-to-end reference

For the established prompt/model/configuration, CUDA must reproduce the CPU
greedy token sequence:

```text
[11576, 42740, 11, 358]
```

The reference is recorded with the native int4 profile
([PROJECTION_FUSION.md, lines
425-434](PROJECTION_FUSION.md#L425-L434)).

Validate:

1. prefill logits parity;
2. each of four decode-step logits;
3. exact greedy tokens;
4. present/KV values for a small context;
5. reset and regenerate exactness;
6. rewind and branch exactness.

### 8.4 Residency and complexity assertions

Add counters/tests that prove:

- initializer H2D uploads happen once per session;
- no initializer H2D occurs during decode;
- no KV D2H/H2D occurs after allocation;
- KV pointers remain stable across steps;
- KV append writes one token slice, not the complete cache;
- per-step host transfer bytes do not grow with context;
- device allocation count is zero during captured replay;
- one capture occurs per generation and later steps replay it;
- separate live generations use distinct capture ids.

### 8.5 Performance validation

Report separately:

- uncaptured CUDA prefill;
- uncaptured CUDA one-token decode;
- captured CUDA one-token decode;
- H2D, kernel, D2H, and CPU-sampling time;
- tokens/s at context lengths 1, 128, 1K, 4K, and the selected maximum;
- peak VRAM and persistent KV/weight allocation.

The first performance gate is “faster than the native CPU path.” The strategic
gate is a like-for-like comparison against llama.cpp CUDA with the same model,
quantization, prompt, context, and sampling. vLLM-class throughput requires
later batching/paging work and is not an M1-M4 acceptance criterion.

## 9. Phased implementation plan

Each milestone is independently reviewable and mergeable. No milestone should
mix executor polymorphism, KV redesign, graph capture, and int4 tuning into one
change.

| Milestone | Deliverable | Acceptance gate | Rough effort | Risk |
|---|---|---|---:|---|
| **M1 — EP-polymorphic executor skeleton** | Change `Executor`/kernel cache/session construction to `Arc<dyn ExecutionProvider>`; add host upload/download trait methods; make buffer device ids correct; upload CUDA initializers; hard-reject host-only paths; add one CUDA single-op session smoke test. Do not wire native generation yet. | CPU suite unchanged; CUDA `Add` or `MatMul` graph runs through `InferenceSession` with H2D input and D2H output; no host dereference of a CUDA pointer. | 3-5 engineer-days | Medium |
| **M2 — Full target decode-step coverage** | Select CUDA from `DevicePreference`; remove the native adapter's CUDA bail-out; add packed-QKV CUDA GQA; register standard-domain `SimplifiedLayerNormalization`; add CUDA SiLU or an EP-aware rewrite; verify all 299 target nodes after optimization; keep weights/activations on device; initially use ordinary dynamic present outputs if needed. | Qwen model loads and produces CPU-parity logits/tokens on CUDA with zero CPU op fallback. Exact tokens `[11576,42740,11,358]`. | 1-2 weeks | High |
| **M3 — Device-resident O(1) KV** | Add external device input/output bindings; allocate fixed-capacity per-layer KV; alias past/present; change GQA to append one token and read only valid prefix; replace context-sized host mask work with fixed buffer/scalars; cursor-only reset/rewind. | No KV PCIe transfer after setup; stable pointers; per-token cache-update work independent of context length; exact token parity. | 1-2 weeks | High |
| **M4 — CUDA graph capture/replay** | First create a real non-default session stream and bind cuDNN/cuBLAS to it; then extend `CudaRuntime` with graph lifecycle and a deliberate serialized graph-ownership design; retain `capture.rs` all-kernel gate; prewarm NVRTC/library state; remove per-call alloc/free/D2H/sync from capture path; persistent I/O; unique capture ids; reset/rewind invalidation. | One capture then replay; zero allocations in replay; changing token/position changes output; two generations do not collide; measurable launch-overhead reduction. | 1-2 weeks | High |
| **M5 — Performance tuning** | Direct packed-int4 `M=1` kernel, persistent workspaces/metadata, fused SiLU/SwiGLU where profitable, faster GQA/SDPA, async/pinned transfers, activation arena, timeline-guided fusion. | Beat native CPU decisively; establish/close a measured gap to llama.cpp CUDA; no regression in parity or memory bounds. | 1-3 weeks, iterative | High |

### 9.1 Review boundaries

- **M1 PR:** EP abstraction and transfers only.
- **M2 PRs:** one target incompatibility per PR where practical
  (packed GQA, SiLU/rewrite, target integration).
- **M3 PR:** KV binding contract first, then GQA append implementation.
- **M4 PRs:** CUDA graph runtime handles, kernel capture audit, decode integration.
- **M5 PRs:** one measured optimization per PR, with before/after profiles.

## 10. Risks and open questions

### 10.1 Decisions needed before implementation

1. **Approve dynamic dispatch.** Recommendation: `Arc<dyn ExecutionProvider>`,
   not generic `Executor<EP>` and not an enum.
2. **Choose the first GPU floor.** `CudaRuntime` currently asks NVRTC for
   `compute_90`, explicitly targeting Hopper ([runtime.rs, lines
   115-145](../crates/onnx-runtime-ep-cuda/src/runtime.rs#L115-L145)). Decide
   whether M1-M5 target H100/H200 only or must support Ampere/Ada at launch.
3. **Choose KV capacity.** 4K costs 96 MiB and 32K costs 768 MiB for this model's
   f32 KV. Decide the default and whether allocation is eager or request-scoped.
4. **Confirm hard-fail CUDA policy.** Recommendation: no transparent CPU fallback
   in native GPU decode.
5. **Choose cudarc graph ownership strategy.** The crate requests cudarc `0.19` with the
   CUDA 13 dynamic-loading feature set ([CUDA Cargo.toml, lines
   17-29](../crates/onnx-runtime-ep-cuda/Cargo.toml#L17-L29)); the lockfile
   resolves `0.19.8` ([Cargo.lock, lines
   563-566](../Cargo.lock#L563-L566)). The basic lifecycle is already available:
   `CudaStream::begin_capture`/`end_capture` and
   `CudaGraph::launch`/`upload`. The remaining issues are selecting the required
   non-default stream and choosing a thread-safe graph ownership model, not API
   absence.

### 10.2 Technical risks

#### Packed-QKV GQA semantics

The target's packed layout must be decoded in exact `Q | K | V` order with
different Q and KV widths. The current kernel's unpacked-only assumptions are
load-bearing. Add independent slicing tests before end-to-end use.

#### Dynamic logical length versus physical capacity

The ONNX graph exposes dynamic past/present shapes, while captured decode needs
fixed physical buffers. The binding layer must not conflate logical tensor shape
with allocation capacity. This is the central M3 API design problem.

#### Capture metadata is currently optimistic

Some kernels return `cuda_graph_compatible=true` but synchronize after every
launch. Capture eligibility must be validated against actual driver behavior,
not trusted by annotation alone.

#### Existing GQA is O(context)

Device residency alone is insufficient. Rebuilding the full present cache each
token preserves PCIe locality but still scales with context. The append kernel
and alias binding are mandatory.

#### Existing int4 kernel is a correctness baseline

Full f32 dequantization per projection per token is unlikely to approach
llama.cpp. Do not mistake M2 end-to-end execution for the performance
architecture. M5's direct packed-int4 path is load-bearing for the north-star.

#### Full-logits D2H

The target's vocabulary is 151,936, so f32 logits are about 594 KiB/token.
That is bounded and acceptable for the first CPU-sampling design, but it may
become visible after graph capture. GPU top-k/sampling is a later optimization.

#### Error handling and fences

`Fence` is currently only an id, and CUDA `copy_async` performs a synchronous
D2D copy ([provider.rs, lines
220-229](../crates/onnx-runtime-ep-cuda/src/provider.rs#L220-L229)). M1 should be
synchronous and correct. M5 may introduce real events/fences, but ownership and
stream ordering need a separate review.

#### Build and deployment compatibility

The CUDA crate dynamically loads driver, cuBLASLt, cuDNN, and NVRTC. M1 must
preserve CPU-only builds and return an actionable “CUDA unavailable” error
rather than changing default CPU behavior.

## 11. Recommended decision

Approve the five-milestone plan with these initial choices:

- `Arc<dyn ExecutionProvider>` for the executor seam;
- Hopper (H100/H200) as the first validated target unless broader coverage is a
  launch requirement;
- hard-fail all-CUDA decode placement;
- 4K default KV capacity for the first Qwen benchmark, configurable upward;
- current cudarc pin retained for M1; M4 must use a non-default stream and choose
  a serialized graph-ownership model;
- packed-QKV GQA, standard-domain `SimplifiedLayerNormalization`, and fused CUDA
  SiLU treated as M2 prerequisites;
- true packed-int4 decode kernel treated as the primary M5 performance item.

After approval, implement M1 alone and review CPU preservation plus the
single-op CUDA session smoke before beginning target-model wiring.
