//! The [`CpuExecutionProvider`]: a host execution provider backed by pure-Rust
//! reference kernels (`docs/architecture/ORT2.md` §4.4).
//!
//! # Memory & safety invariants (ep-api safety review)
//!
//! This EP is the allocator/deallocator for every buffer it hands out. It
//! upholds the five must-hold invariants from the ep-api safety review:
//!
//! 1. **View bounds** — kernels only read/write within the extent a
//!    [`TensorView`](onnx_runtime_ep_api::TensorView)'s shape/strides/offset
//!    describe; the caller that owns the backing buffer verifies storage bounds
//!    via [`crate::strided::view_in_bounds`] before dispatch (a `TensorView`
//!    cannot see its allocation size).
//! 2. **Single-free** — every [`allocate`](CpuExecutionProvider::allocate)
//!    pairs with exactly one [`deallocate`](CpuExecutionProvider::deallocate);
//!    `DeviceBuffer` has no `Drop`, so a dropped handle leaks but never
//!    double-frees.
//! 3. **No cross-EP free** — `deallocate`/`copy` assert the buffer's device
//!    matches this EP's device.
//! 4. **`copy` size** — `copy`/`copy_async` reject `size` larger than either
//!    endpoint.
//! 5. **Thread-affine allocators** — N/A: host `malloc` addresses are portable,
//!    so `DeviceBuffer` is soundly `Send`/`Sync` (documented in ep-api).

use onnx_runtime_ep_api::{
    ClaimPreference, Cost, DeviceBuffer, EpConfig, EpError, ExecutionProvider, Fence, Kernel,
    KernelMatch, OpRegistry, Result, deny, structural_input_bytes,
};
use onnx_runtime_ir::{DataType, DeviceId, DeviceType, Node, Shape, TensorLayout};

use crate::WeightOffloadHostCache;
use crate::kernels::{build_cpu_registry, build_cpu_registry_with_weight_offload_cache};
use crate::optimizer::cpu_optimization_passes;

/// CPU execution provider. Always available; the fallback EP for any op.
///
/// Holds the CPU op → kernel-factory registry, built once at construction. The
/// registry is also exposed to the session (Track D) so placement and kernel
/// instantiation share one source of truth.
pub struct CpuExecutionProvider {
    device: DeviceId,
    initialized: bool,
    registry: OpRegistry,
    /// Where this EP's buffers come from.
    ///
    /// The same `DeviceAllocator` contract the ONNX Runtime side uses, so an
    /// allocator a caller writes serves both backends instead of having to be
    /// written twice. Defaults to `HostAllocator`, which is the `std::alloc`
    /// code this EP used to inline.
    memory: std::sync::Arc<dyn onnx_runtime_memory_governor::DeviceAllocator>,
}

impl std::fmt::Debug for CpuExecutionProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpuExecutionProvider")
            .field("device", &self.device)
            .field("initialized", &self.initialized)
            .field("registered_ops", &self.registry.len())
            .finish()
    }
}

impl Default for CpuExecutionProvider {
    fn default() -> Self {
        Self::new()
    }
}

/// The allocator every CPU EP uses unless a caller installs another.
///
/// # Why this is not `HostAllocator`
///
/// Graph outputs are handed to the caller by *moving* the produced buffer out of
/// the executor, which is zero-copy but forfeits that value's cross-run buffer
/// reuse: the next run allocates the output afresh. For an output large enough
/// that glibc serves it with `mmap` — every attention activation and every
/// present-KV tensor in a real model — that means a fresh, demand-zeroed mapping
/// per run, so each page traps and is zeroed by the kernel before the first
/// store to it can retire. Measured on a Whisper cross-attention graph with a
/// 180 MB output, this was exactly one `mmap`/`munmap` pair and 511 minor faults
/// **per run**, against an ONNX Runtime arm that pays neither because its CPU
/// allocator reuses memory it already owns.
///
/// [`LargeAllocCache`] recycles exactly that band and delegates everything
/// smaller to `HostAllocator` untouched, so the small-allocation path this
/// project already measured as best-served-by-the-system-allocator is unchanged.
///
/// Set `ONNX_GENAI_HOST_ALLOC_CACHE_BYTES=0` to get the old behaviour back.
///
/// # Why one cache per EP rather than one per process
///
/// The executor allocates a moved-out output through its own EP and hands that
/// same EP to the resulting tensor, so the matching free comes back to the EP
/// that made the allocation. Scoping the cache to the EP therefore loses no
/// reuse, and it bounds retention by the session's life instead of the
/// process's.
fn default_cpu_memory() -> std::sync::Arc<dyn onnx_runtime_memory_governor::DeviceAllocator> {
    std::sync::Arc::new(onnx_runtime_memory_governor::LargeAllocCache::default())
}

impl CpuExecutionProvider {
    /// Construct a CPU EP bound to `CPU:0` with all Phase-1 kernels registered.
    pub fn new() -> Self {
        Self {
            device: DeviceId::cpu(),
            initialized: false,
            registry: build_cpu_registry(),
            memory: default_cpu_memory(),
        }
    }

    /// Construct a CPU EP whose QMoE kernels share one governor-owned host-cache partition.
    pub fn with_weight_offload_host_cache(host_cache: WeightOffloadHostCache) -> Self {
        Self {
            device: DeviceId::cpu(),
            initialized: false,
            registry: build_cpu_registry_with_weight_offload_cache(host_cache),
            memory: default_cpu_memory(),
        }
    }

    /// Take buffers from `memory` instead of the system allocator.
    ///
    /// The same `DeviceAllocator` a caller installs on the ONNX Runtime side.
    /// That is the point of the contract living in a crate both backends
    /// depend on: an allocator is written once, not once per backend.
    pub fn with_memory(
        mut self,
        memory: std::sync::Arc<dyn onnx_runtime_memory_governor::DeviceAllocator>,
    ) -> Self {
        self.memory = memory;
        self
    }

    /// Construct and initialize a CPU EP with a governor-owned host-cache partition.
    pub fn initialized_with_weight_offload_host_cache(
        host_cache: WeightOffloadHostCache,
    ) -> Result<Self> {
        let mut ep = Self::with_weight_offload_host_cache(host_cache);
        ep.initialize(&Default::default())?;
        Ok(ep)
    }

    /// Borrow the CPU op registry (shared with the session layer).
    pub fn registry(&self) -> &OpRegistry {
        &self.registry
    }
}

impl ExecutionProvider for CpuExecutionProvider {
    fn name(&self) -> &str {
        "cpu_ep"
    }

    fn device_type(&self) -> DeviceType {
        DeviceType::Cpu
    }

    fn device_id(&self) -> DeviceId {
        self.device
    }

    fn initialize(&mut self, _config: &EpConfig) -> Result<()> {
        // Pure-Rust kernels need no device resources or external libraries. This
        // is the earliest per-session hook that runs before any GEMM, so it is
        // where the explicit decode budget is turned into a process-wide bound
        // on prefill/MLAS Rayon parallelism (and, on Linux, CPU affinity) -- a
        // no-op unless a budget is set.
        crate::kernels::matmul_nbits::bound_process_to_decode_budget();
        self.initialized = true;
        Ok(())
    }

    fn shutdown(&mut self) -> Result<()> {
        self.initialized = false;
        Ok(())
    }

    /// Routing preference, kept deliberately apart from [`Self::supports_op`].
    ///
    /// See [`crate::assignment_policy`] for the measured evidence behind each
    /// deferral and for why a performance decline cannot live in `supports_op`.
    fn claim_preference(
        &self,
        op: &Node,
        opset: u64,
        shapes: &[Shape],
        input_dtypes: &[DataType],
    ) -> ClaimPreference {
        crate::assignment_policy::claim_preference(op, opset, shapes, input_dtypes)
    }

    /// Short-circuit the default adapter for the nodes the policy has no
    /// opinion about.
    ///
    /// The default [`ExecutionProvider::claim_preference_node`] deep-clones
    /// every input [`Shape`] before it can ask, and it runs for every node in
    /// the graph. This policy governs five elementwise ops; a transformer graph
    /// is overwhelmingly not those, so the cheap `(domain, op_type)` test comes
    /// first and the allocation only happens when the answer can differ from
    /// [`ClaimPreference::Claim`].
    fn claim_preference_node(
        &self,
        view: &onnx_runtime_ir::GraphView<'_>,
        node: onnx_runtime_ir::NodeIndex,
        opset: u64,
    ) -> ClaimPreference {
        let ir_node = view.node(node);
        if !crate::assignment_policy::governs(ir_node) {
            return ClaimPreference::Claim;
        }
        let inputs = view.node_inputs(node);
        let shapes = inputs
            .iter()
            .map(|input| {
                input
                    .map(|value| view.value(value).shape.clone())
                    .unwrap_or_default()
            })
            .collect::<Vec<_>>();
        let input_dtypes = inputs
            .iter()
            .map(|input| {
                input
                    .map(|value| view.value(value).dtype)
                    .unwrap_or(DataType::Undefined)
            })
            .collect::<Vec<_>>();
        crate::assignment_policy::claim_preference(ir_node, opset, &shapes, &input_dtypes)
    }

    fn supports_op(
        &self,
        op: &Node,
        opset: u64,
        shapes: &[Shape],
        input_dtypes: &[DataType],
        _layouts: &[TensorLayout],
    ) -> KernelMatch {
        // Keyed on (op_type, domain, opset) via the registry — the single source
        // of truth for "is this operator version supported". This
        // accepts standard default-domain (`""`/`ai.onnx`) ops and any contrib
        // ops (e.g. fused `com.microsoft` ops) the registry knows, without a
        // hardcoded op/domain whitelist.
        let domain = if op.domain.is_empty() {
            "ai.onnx"
        } else {
            &op.domain
        };
        if !self.registry.supports(&op.op_type, &op.domain, opset) {
            if let Some(since) = self
                .registry
                .earliest_since_version(&op.op_type, &op.domain)
            {
                deny!(
                    "no handler for {}::{} at opset {} — this EP registers {} since opset {} (or: add a claim+handler)",
                    domain,
                    op.op_type,
                    opset,
                    op.op_type,
                    since
                );
            }
            deny!(
                "no handler for {}::{} at opset {} — add a claim+handler",
                domain,
                op.op_type,
                opset
            );
        }
        if op.op_type == "CompressedSparseAttention"
            && op.domain == "pkg.nxrt"
            && let Some(reason) = crate::kernels::compressed_sparse_attention::unsupported_reason(
                op,
                shapes,
                input_dtypes,
            )
        {
            return KernelMatch::unsupported(reason);
        }
        if op.op_type == "BlockQuantizedMoE"
            && op.domain == "pkg.nxrt"
            && let Some(reason) =
                crate::kernels::block_quantized_moe::unsupported_reason(op, shapes, input_dtypes)
        {
            return KernelMatch::unsupported(reason);
        }
        if op.op_type == "IndexShare"
            && op.domain == "pkg.nxrt"
            && let Some(reason) =
                crate::kernels::index_share::unsupported_reason(op, shapes, input_dtypes)
        {
            return KernelMatch::unsupported(reason);
        }
        if op.op_type == "VarlenAttention"
            && op.domain == "pkg.nxrt"
            && let Some(reason) =
                crate::kernels::varlen_attention::unsupported_reason(op, shapes, input_dtypes)
        {
            return KernelMatch::unsupported(reason);
        }
        if op.op_type == "PackedVarlenAttention"
            && op.domain == "pkg.nxrt"
            && let Some(reason) = crate::kernels::packed_varlen_attention::unsupported_reason(
                op,
                shapes,
                input_dtypes,
            )
        {
            return KernelMatch::unsupported(reason);
        }
        if op.op_type == "PackedMultiHeadAttention"
            && op.domain == "com.microsoft"
            && let Some(reason) = crate::kernels::packed_multi_head_attention::unsupported_reason(
                op,
                shapes,
                input_dtypes,
            )
        {
            return KernelMatch::unsupported(reason);
        }
        if op.op_type == "ScatterND"
            && (op.domain.is_empty() || op.domain == "ai.onnx")
            && let Some(reason) =
                crate::kernels::indexing::scatter_nd_unsupported_reason(input_dtypes)
        {
            return KernelMatch::unsupported(reason);
        }
        if op.op_type == "QLinearMatMul"
            && (op.domain.is_empty() || op.domain == "ai.onnx")
            && let Some(reason) =
                crate::kernels::qlinear_matmul::unsupported_reason(input_dtypes, shapes)
        {
            return KernelMatch::unsupported(reason);
        }
        // The reference kernels produce contiguous row-major outputs and accept
        // strided inputs, so no input layout is required.
        let output_layouts = vec![TensorLayout::contiguous(); op.outputs.len()];
        // Report *structure only*, never a machine rate (issue #995). The EP
        // knows how many bytes this node reads from its real dtypes and shapes;
        // it does NOT know this host's memory bandwidth, FLOP/s, or launch
        // latency, so it must not invent a per-element time. The old
        // `Cost::new(elems, elems, 0.0).with_bytes_moved(elems*4)` fabricated
        // both: a CPU-is-100×-slower-than-CUDA constant and an f32 byte count
        // (wrong by 8× for int4 weights). Time components are left zero — the
        // placement cost model (`onnx-runtime-cost-model`) divides this
        // structural `bytes_moved` by the host's *measured* rate. `bytes_moved`
        // stays monotonic in problem size so any interim consumer still prefers
        // a smaller op over a larger one.
        let bytes_moved = structural_input_bytes(shapes, input_dtypes);
        let cost = Cost::ZERO.with_bytes_moved(bytes_moved);
        KernelMatch::Supported {
            cost,
            required_input_layouts: None,
            output_layouts,
        }
    }

    fn get_kernel(&self, op: &Node, shapes: &[Vec<usize>], opset: u64) -> Result<Box<dyn Kernel>> {
        // Select the highest registered `since_version` that is <= the graph's
        // effective opset for this op's domain. Ops with a single registration
        // (since_version 1) always match; opset-specialized ops (e.g. Softmax,
        // registered at both 1 and 13) get the version-correct kernel.
        let factory = self
            .registry
            .lookup(&op.op_type, &op.domain, opset)
            .ok_or_else(|| EpError::NoEpForOp {
                domain: if op.domain.is_empty() {
                    "ai.onnx".to_string()
                } else {
                    op.domain.clone()
                },
                op_type: op.op_type.clone(),
                opset,
            })?;
        factory.create(op, shapes)
    }

    fn custom_passes(&self) -> Vec<Box<dyn onnx_runtime_optimizer::OptimizationPass>> {
        cpu_optimization_passes()
    }

    fn allocate(&self, size: usize, alignment: usize) -> Result<DeviceBuffer> {
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(EpError::AlignmentError);
        }
        // One allocator for the whole project: the same `DeviceAllocator` a
        // caller can install on the ONNX Runtime side backs this one, so an
        // allocator is written once rather than once per backend. The default
        // is `HostAllocator`, which is the `std::alloc` code this used to
        // inline.
        let ptr = self.memory.allocate(size, alignment).map_err(|error| {
            // Keep what the allocator said. Reporting every failure as "out of
            // memory" describes an alignment rejection, or a substituted
            // allocator refusing for its own reason, as exhausted RAM and sends
            // the reader looking in the wrong place.
            EpError::KernelFailed(format!(
                "cpu_ep: could not allocate {size} bytes aligned to {alignment}: {error}"
            ))
        })?;
        // SAFETY: `ptr` is a fresh, unique, non-null allocation of at least
        // `size` bytes aligned to `alignment`, owned by this EP and freed
        // exactly once in `deallocate` (invariant #2). No other handle aliases
        // it. We record the caller-requested `size`.
        Ok(unsafe {
            DeviceBuffer::from_raw_parts(ptr.as_ptr().cast(), self.device, size, alignment)
        })
    }

    fn deallocate(&self, buffer: DeviceBuffer) -> Result<()> {
        // Invariant #3: never free a buffer that belongs to another EP/device.
        assert_eq!(
            buffer.device(),
            self.device,
            "cpu_ep: refusing to deallocate a buffer from device {:?}",
            buffer.device()
        );
        // Borrowed buffers alias foreign memory (e.g. an mmap'd weight file)
        // that this EP never allocated — freeing it would be undefined
        // behavior. The real owner outlives the buffer and frees it itself.
        if buffer.is_borrowed() {
            return Ok(());
        }
        let size = buffer.len();
        let align = buffer.alignment();
        let ptr = buffer.into_raw() as *mut u8;
        let Some(ptr) = std::ptr::NonNull::new(ptr) else {
            return Ok(());
        };
        // SAFETY: `ptr`, `size` and `align` are the triple this EP obtained
        // from `self.memory` in `allocate` (invariant #2); `into_raw` consumed
        // the owning handle so no alias remains, and this is the single free of
        // that allocation.
        unsafe { self.memory.deallocate(ptr, size, align) };
        Ok(())
    }

    fn copy(&self, src: &DeviceBuffer, dst: &mut DeviceBuffer, size: usize) -> Result<()> {
        // Invariant #3: both endpoints must belong to this EP.
        assert_eq!(
            src.device(),
            self.device,
            "cpu_ep::copy: foreign src buffer"
        );
        assert_eq!(
            dst.device(),
            self.device,
            "cpu_ep::copy: foreign dst buffer"
        );
        // Invariant #4: never read/write past either endpoint.
        if size > src.len() || size > dst.len() {
            return Err(EpError::KernelFailed(format!(
                "cpu_ep::copy: size {size} exceeds src {} or dst {}",
                src.len(),
                dst.len()
            )));
        }
        if size == 0 {
            return Ok(());
        }
        let src_ptr = src.as_ptr() as *const u8;
        let dst_ptr = dst.as_mut_ptr() as *mut u8;
        // SAFETY: both pointers are valid host allocations of at least `size`
        // bytes (checked above). They name distinct `DeviceBuffer`s — `dst` is
        // borrowed `&mut`, so it cannot alias `src` — hence non-overlapping.
        unsafe {
            std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, size);
        }
        Ok(())
    }

    fn copy_async(&self, src: &DeviceBuffer, dst: &mut DeviceBuffer, size: usize) -> Result<Fence> {
        // Host copies are synchronous; perform it and return a signaled fence.
        self.copy(src, dst, size)?;
        Ok(Fence::default())
    }

    fn sync(&self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ep_api::abi::OrtGraphView;
    use onnx_runtime_ir::{Attribute, FrozenGraph, Graph, NodeId, static_shape};
    use std::ffi::c_void;

    fn stateful_csa_node(ratio: i64, input_count: usize, output_count: usize) -> Node {
        let mut graph = Graph::new();
        let inputs = (0..input_count)
            .map(|index| {
                Some(graph.create_named_value(
                    format!("input_{index}"),
                    DataType::Float32,
                    static_shape([]),
                ))
            })
            .collect();
        let outputs = (0..output_count)
            .map(|index| {
                graph.create_named_value(
                    format!("output_{index}"),
                    DataType::Float32,
                    static_shape([]),
                )
            })
            .collect();
        let mut node = Node::new(NodeId(0), "CompressedSparseAttention", inputs, outputs);
        node.domain = "pkg.nxrt".into();
        node.attributes
            .insert("num_heads".into(), Attribute::Int(1));
        node.attributes
            .insert("head_dim".into(), Attribute::Int(512));
        node.attributes
            .insert("qk_rope_head_dim".into(), Attribute::Int(64));
        node.attributes
            .insert("compression_ratio".into(), Attribute::Int(ratio));
        if ratio == 4 {
            node.attributes
                .insert("index_num_heads".into(), Attribute::Int(1));
            node.attributes
                .insert("index_head_dim".into(), Attribute::Int(128));
            node.attributes
                .insert("index_topk".into(), Attribute::Int(1));
        }
        node
    }

    #[test]
    fn identity_and_lifecycle() {
        let mut ep = CpuExecutionProvider::new();
        assert_eq!(ep.name(), "cpu_ep");
        assert_eq!(ep.device_type(), DeviceType::Cpu);
        assert_eq!(ep.device_id(), DeviceId::cpu());
        ep.initialize(&EpConfig::default()).unwrap();
        assert!(ep.initialized);
        ep.shutdown().unwrap();
        assert!(!ep.initialized);
    }

    #[test]
    fn graph_view_capabilities_follow_the_cpu_kernel_registry() {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 17);
        let input = graph.create_value(DataType::Float32, static_shape([2]));
        let add_out = graph.create_value(DataType::Float32, static_shape([2]));
        let custom_out = graph.create_value(DataType::Float32, static_shape([2]));
        let relu_out = graph.create_value(DataType::Float32, static_shape([2]));
        graph.add_input(input);
        let add = graph.insert_node(Node::new(
            NodeId(0),
            "Add",
            vec![Some(input), Some(input)],
            vec![add_out],
        ));
        let custom = graph.insert_node(Node::new(
            NodeId(0),
            "UnregisteredCustom",
            vec![Some(add_out)],
            vec![custom_out],
        ));
        let relu = graph.insert_node(Node::new(
            NodeId(0),
            "Relu",
            vec![Some(custom_out)],
            vec![relu_out],
        ));
        graph.add_output(relu_out);

        let frozen = FrozenGraph::build(graph).unwrap();
        let view = frozen.view();
        let ep = CpuExecutionProvider::new();
        let claims = OrtGraphView::new(&view).query_capabilities(&ep);
        assert_eq!(claims.len(), 2);
        assert_eq!(claims[0].node_ids, vec![add]);
        assert_eq!(claims[1].node_ids, vec![relu]);
        assert!(ep.registry().supports("Add", "", 17));
        assert!(!ep.registry().supports("UnregisteredCustom", "", 17));
        assert!(ep.registry().supports("Relu", "", 17));
        assert!(!claims.iter().any(|claim| claim.node_ids.contains(&custom)));
    }

    #[test]
    fn allocate_deallocate_single_free_and_aligned() {
        let ep = CpuExecutionProvider::new();
        let buf = ep.allocate(256, 64).unwrap();
        assert_eq!(buf.len(), 256);
        assert_eq!(buf.alignment(), 64);
        assert_eq!(buf.device(), DeviceId::cpu());
        // 64-byte aligned base.
        assert_eq!(buf.as_ptr() as usize % 64, 0);
        // Single free — a double free would trip ASan/Miri.
        ep.deallocate(buf).unwrap();
    }

    #[test]
    fn allocate_zero_size_is_nonnull() {
        let ep = CpuExecutionProvider::new();
        let buf = ep.allocate(0, 16).unwrap();
        assert_eq!(buf.len(), 0);
        assert!(!buf.as_ptr().is_null());
        ep.deallocate(buf).unwrap();
    }

    /// `deallocate` must be a no-op free for a borrowed buffer: it aliases
    /// memory the EP never allocated (here a `Vec`), so freeing it would be UB.
    /// After deallocation the backing must remain fully valid.
    #[test]
    fn deallocate_borrowed_buffer_is_a_noop_free() {
        let ep = CpuExecutionProvider::new();
        let mut backing = vec![42u8; 128];
        let ptr = backing.as_mut_ptr() as *mut c_void;
        // SAFETY: `ptr`/`len` name `backing`'s live allocation; `backing`
        // outlives the buffer, we never write through it, and `deallocate` must
        // skip the free because the buffer is borrowed.
        let buf = unsafe { DeviceBuffer::from_borrowed_parts(ptr, ep.device_id(), 128, 1) };
        assert!(buf.is_borrowed());
        ep.deallocate(buf).unwrap();
        // No free happened: the Vec is still valid and unmodified.
        assert!(backing.iter().all(|&b| b == 42));
        backing[0] = 1; // proves the allocation is live (would be UAF if freed)
        assert_eq!(backing[0], 1);
    }

    #[test]
    fn allocate_rejects_bad_alignment() {
        let ep = CpuExecutionProvider::new();
        assert!(matches!(ep.allocate(16, 0), Err(EpError::AlignmentError)));
        assert!(matches!(
            ep.allocate(16, 24), // not a power of two
            Err(EpError::AlignmentError)
        ));
    }

    #[test]
    fn copy_moves_bytes_and_checks_size() {
        let ep = CpuExecutionProvider::new();
        let mut src = ep.allocate(16, 16).unwrap();
        let mut dst = ep.allocate(16, 16).unwrap();
        // Write a pattern into src.
        // SAFETY: host buffer of 16 bytes, unique &mut access.
        unsafe {
            let p = src.as_mut_ptr() as *mut u8;
            for i in 0..16u8 {
                *p.add(i as usize) = i;
            }
        }
        ep.copy(&src, &mut dst, 16).unwrap();
        // SAFETY: dst is a valid 16-byte host buffer.
        unsafe {
            let p = dst.as_ptr() as *const u8;
            for i in 0..16u8 {
                assert_eq!(*p.add(i as usize), i);
            }
        }
        // Oversized copy is rejected.
        assert!(ep.copy(&src, &mut dst, 32).is_err());
        ep.deallocate(src).unwrap();
        ep.deallocate(dst).unwrap();
    }

    #[test]
    fn deallocate_rejects_cross_device_buffer() {
        let ep = CpuExecutionProvider::new();
        // Fabricate a buffer tagged with a CUDA device to trip invariant #3.
        let boxed = vec![0u8; 8].into_boxed_slice();
        let ptr = Box::into_raw(boxed) as *mut c_void;
        // SAFETY: valid 8-byte host allocation; we only use it to exercise the
        // device assert. The allocation is reclaimed after catching the panic.
        let foreign = unsafe { DeviceBuffer::from_raw_parts(ptr, DeviceId::cuda(0), 8, 8) };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = ep.deallocate(foreign); // must panic before freeing
        }));
        let panic = result.expect_err("cross-device deallocate must panic before freeing");
        let message = if let Some(message) = panic.downcast_ref::<String>() {
            message.as_str()
        } else if let Some(message) = panic.downcast_ref::<&str>() {
            *message
        } else {
            panic!("cross-device deallocate panic used a non-string payload");
        };
        assert!(
            message.contains("cpu_ep: refusing to deallocate a buffer from device"),
            "unexpected deallocate panic: {message}"
        );
        assert!(
            message.contains("Cuda") || message.contains("cuda"),
            "cross-device deallocate panic did not identify the foreign CUDA device: {message}"
        );
        // SAFETY: `deallocate` panicked before freeing, so `ptr` still names the
        // original boxed slice allocation.
        unsafe {
            drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
                ptr as *mut u8,
                8,
            )));
        }
    }

    #[test]
    fn get_kernel_dispatches_phase1_ops() {
        let ep = CpuExecutionProvider::new();
        for (i, op) in crate::kernels::PHASE1_OPS.iter().enumerate() {
            let mut node = Node::new(onnx_runtime_ir::NodeId(i as u32), *op, vec![], vec![]);
            if *op == "BitShift" {
                node.attributes
                    .insert("direction".into(), Attribute::String(b"RIGHT".to_vec()));
            }
            assert!(ep.get_kernel(&node, &[], 17).is_ok(), "no kernel for {op}");
        }
        let bad = Node::new(onnx_runtime_ir::NodeId(99), "UnknownOp", vec![], vec![]);
        assert!(ep.get_kernel(&bad, &[], 17).is_err());
    }

    #[test]
    fn supports_op_reflects_selected_operator_groups() {
        let ep = CpuExecutionProvider::new();
        let mm = Node::new(onnx_runtime_ir::NodeId(0), "MatMul", vec![], vec![]);
        assert!(ep.supports_op(&mm, 17, &[], &[], &[]).is_supported());
        let conv = Node::new(onnx_runtime_ir::NodeId(1), "Conv", vec![], vec![]);
        assert_eq!(
            ep.supports_op(&conv, 17, &[], &[], &[]).is_supported(),
            cfg!(feature = "ops-cnn")
        );
    }

    #[test]
    fn supports_op_is_opset_aware_for_standard_gelu() {
        let ep = CpuExecutionProvider::new();
        let gelu = Node::new(onnx_runtime_ir::NodeId(0), "Gelu", vec![], vec![]);

        let rejected = ep.supports_op(&gelu, 19, &[], &[], &[]);
        let reason = rejected.reason().expect("opset 19 must be declined");
        assert!(
            reason.contains("no handler for ai.onnx::Gelu at opset 19"),
            "{reason}"
        );
        assert!(reason.contains("registers Gelu since opset 20"), "{reason}");

        assert!(ep.supports_op(&gelu, 20, &[], &[], &[]).is_supported());
    }

    #[test]
    fn qlinear_matmul_claim_gate_checks_quantized_dtypes() {
        let ep = CpuExecutionProvider::new();
        let node = Node::new(onnx_runtime_ir::NodeId(0), "QLinearMatMul", vec![], vec![]);
        let valid = [
            DataType::Uint8,
            DataType::Float32,
            DataType::Uint8,
            DataType::Int8,
            DataType::Float32,
            DataType::Int8,
            DataType::Float32,
            DataType::Uint8,
        ];
        assert!(
            ep.supports_op(&node, 10, &[], &valid, &[]).is_supported(),
            "valid QLinearMatMul types should be claimed"
        );

        let mut invalid = valid;
        invalid[0] = DataType::Float32;
        let rejected = ep.supports_op(&node, 10, &[], &invalid, &[]);
        assert!(!rejected.is_supported());
        let reason = rejected.reason().unwrap();
        assert!(reason.contains("A must have Int8 or Uint8"), "{reason}");

        let valid_shapes = [
            static_shape([2, 3, 4]),
            static_shape([2, 3, 1]),
            static_shape([2, 3, 1]),
            static_shape([1, 4, 5]),
            static_shape([1, 1, 5]),
            static_shape([1, 1, 5]),
            static_shape([]),
            static_shape([]),
        ];
        assert!(
            ep.supports_op(&node, 10, &valid_shapes, &valid, &[])
                .is_supported(),
            "valid N-D per-row/per-column quantization should be claimed"
        );
        let valid_2d_shapes = [
            static_shape([3, 4]),
            static_shape([3]),
            static_shape([3]),
            static_shape([4, 5]),
            static_shape([5]),
            static_shape([5]),
            static_shape([1]),
            static_shape([1]),
        ];
        assert!(
            ep.supports_op(&node, 10, &valid_2d_shapes, &valid, &[])
                .is_supported(),
            "valid 2-D per-row/per-column quantization should be claimed"
        );

        let mut mismatched_pair = valid_shapes.clone();
        mismatched_pair[2] = static_shape([]);
        let rejected = ep.supports_op(&node, 10, &mismatched_pair, &valid, &[]);
        assert!(!rejected.is_supported());
        assert!(
            rejected.reason().unwrap().contains("shapes must match"),
            "{:?}",
            rejected.reason()
        );

        let mut invalid_nd_vector = valid_shapes;
        invalid_nd_vector[1] = static_shape([3]);
        invalid_nd_vector[2] = static_shape([3]);
        let rejected = ep.supports_op(&node, 10, &invalid_nd_vector, &valid, &[]);
        assert!(!rejected.is_supported());
        assert!(
            rejected.reason().unwrap().contains("invalid a"),
            "{:?}",
            rejected.reason()
        );
    }

    #[test]
    fn scatter_nd_claim_gate_only_accepts_executable_dtypes() {
        let ep = CpuExecutionProvider::new();
        let node = Node::new(onnx_runtime_ir::NodeId(0), "ScatterND", vec![], vec![]);

        for dtype in [
            DataType::Float32,
            DataType::Float16,
            DataType::BFloat16,
            DataType::Float64,
            DataType::Int8,
            DataType::Int16,
            DataType::Int32,
            DataType::Int64,
            DataType::Uint8,
            DataType::Uint16,
            DataType::Uint32,
            DataType::Uint64,
        ] {
            assert!(
                ep.supports_op(&node, 18, &[], &[dtype, DataType::Int64, dtype], &[])
                    .is_supported(),
                "{dtype:?} should be claimed"
            );
        }

        for dtype in [
            DataType::String,
            DataType::Bool,
            DataType::Complex64,
            DataType::Complex128,
        ] {
            let rejected = ep.supports_op(&node, 18, &[], &[dtype, DataType::Int64, dtype], &[]);
            assert!(!rejected.is_supported(), "{dtype:?} must not be claimed");
            assert!(
                rejected.reason().unwrap().contains("not implemented"),
                "{dtype:?}"
            );
        }
    }

    #[test]
    fn supports_op_rejects_malformed_csa_ratio_specific_arity() {
        let ep = CpuExecutionProvider::new();
        let mut ratio4_missing_index = stateful_csa_node(4, 19, 5);
        ratio4_missing_index.inputs[17] = None;
        for (node, expected) in [
            (
                ratio4_missing_index,
                "ratio-4 requires all eight positional index inputs (11..=18)",
            ),
            (
                stateful_csa_node(4, 19, 4),
                "ratio-4 requires 5 or 6 outputs, got 4",
            ),
            (
                stateful_csa_node(128, 12, 3),
                "ratio-4-only inputs (11..=18)",
            ),
            (
                stateful_csa_node(128, 11, 4),
                "ratio-128 supports exactly 3 outputs, got 4",
            ),
        ] {
            let rejected = ep.supports_op(&node, 1, &[], &[], &[]);
            assert!(!rejected.is_supported());
            let reason = rejected.reason().expect("CSA claim must be denied");
            assert!(reason.contains(expected), "{reason}");
        }
    }

    #[test]
    fn supports_fused_contrib_domain_layernorm() {
        let ep = CpuExecutionProvider::new();
        // The optimizer emits fused LayerNormalization in `com.microsoft`; the
        // EP must accept it (bound to the same kernel as the standard op).
        let mut fused = Node::new(
            onnx_runtime_ir::NodeId(0),
            "LayerNormalization",
            vec![],
            vec![],
        );
        fused.domain = "com.microsoft".to_string();
        assert!(ep.supports_op(&fused, 1, &[], &[], &[]).is_supported());
        assert!(ep.get_kernel(&fused, &[], 1).is_ok());

        // The fused `FusedMatMulBias` (MatMul+Add) now has a contrib-domain
        // kernel, so it is supported and instantiable.
        let mut fmb = Node::new(
            onnx_runtime_ir::NodeId(1),
            "FusedMatMulBias",
            vec![],
            vec![],
        );
        fmb.domain = "com.microsoft".to_string();
        assert!(ep.supports_op(&fmb, 1, &[], &[], &[]).is_supported());
        assert!(ep.get_kernel(&fmb, &[], 1).is_ok());

        // The fused `FusedGemm` (MatMul+Add+Relu) now has a contrib-domain
        // kernel too, so it is supported and instantiable.
        let mut fg = Node::new(onnx_runtime_ir::NodeId(2), "FusedGemm", vec![], vec![]);
        fg.domain = "com.microsoft".to_string();
        assert!(ep.supports_op(&fg, 1, &[], &[], &[]).is_supported());
        assert!(ep.get_kernel(&fg, &[], 1).is_ok());

        // The fused `FusedAttention` (SDPA core) is supported in the contrib
        // domain; its factory needs the synthesized `scale` attribute to
        // instantiate.
        let mut fa = Node::new(onnx_runtime_ir::NodeId(4), "FusedAttention", vec![], vec![]);
        fa.domain = "com.microsoft".to_string();
        assert!(ep.supports_op(&fa, 1, &[], &[], &[]).is_supported());
        fa.attributes
            .insert("scale".to_string(), onnx_runtime_ir::Attribute::Float(0.5));
        assert!(ep.get_kernel(&fa, &[], 1).is_ok());

        // A contrib op with no kernel is still rejected — support is keyed on
        // (op_type, domain).
        let mut unknown = Node::new(
            onnx_runtime_ir::NodeId(3),
            "NotARealFusedOp",
            vec![],
            vec![],
        );
        unknown.domain = "com.microsoft".to_string();
        assert!(!ep.supports_op(&unknown, 1, &[], &[], &[]).is_supported());
    }
}
