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
    Cost, DeviceBuffer, EpConfig, EpError, ExecutionProvider, Fence, Kernel, KernelMatch,
    OpRegistry, Result, deny, structural_input_bytes,
};
use onnx_runtime_ir::{DataType, DeviceId, DeviceType, Node, Shape, TensorLayout};

use crate::WeightOffloadHostCache;
use crate::kernels::{
    build_cpu_registry, build_cpu_registry_with_weight_offload_cache,
    build_cpu_registry_with_weight_offload_cache_and_einsum_retention,
};
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

    /// Construct a CPU EP with one immutable Einsum scratch-retention verdict.
    pub fn with_einsum_scratch_retention(
        einsum_scratch_retention: crate::kernels::einsum::EinsumScratchRetention,
    ) -> Self {
        Self::with_weight_offload_host_cache_and_einsum_retention(
            crate::kernels::qmoe::default_weight_offload_host_cache().clone(),
            einsum_scratch_retention,
        )
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

    /// Construct a CPU EP whose compiled Einsum kernels retain scratch only
    /// when this provider/session's immutable memory-plan verdict admitted it.
    pub fn with_weight_offload_host_cache_and_einsum_retention(
        host_cache: WeightOffloadHostCache,
        einsum_scratch_retention: crate::kernels::einsum::EinsumScratchRetention,
    ) -> Self {
        Self {
            device: DeviceId::cpu(),
            initialized: false,
            registry: build_cpu_registry_with_weight_offload_cache_and_einsum_retention(
                host_cache,
                einsum_scratch_retention,
            ),
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

    /// Construct and initialize a CPU EP with session-owned Einsum scratch
    /// retention.
    pub fn initialized_with_weight_offload_host_cache_and_einsum_retention(
        host_cache: WeightOffloadHostCache,
        einsum_scratch_retention: crate::kernels::einsum::EinsumScratchRetention,
    ) -> Result<Self> {
        let mut ep = Self::with_weight_offload_host_cache_and_einsum_retention(
            host_cache,
            einsum_scratch_retention,
        );
        ep.initialize(&Default::default())?;
        Ok(ep)
    }

    /// Borrow the CPU op registry (shared with the session layer).
    pub fn registry(&self) -> &OpRegistry {
        &self.registry
    }
}

/// Smallest host copy worth splitting across threads.
///
/// Calibrated **in the session**, not in a micro-benchmark. A loop that copies
/// the same buffer repeatedly measures an L3-resident copy and says the
/// crossover is tens of MiB; the session copies a graph input that the previous
/// run's output and intermediates have already evicted, so the read comes from
/// DRAM and parallelises far earlier. Trusting the micro-benchmark set this
/// threshold 6x too high and left a 3.3x win on the floor.
///
/// Measured with `bench_generic --phase-profile`, `run_scoped.bind_inputs`
/// microseconds per run, 25 runs after 10 warmups, 16-thread decode budget,
/// median of 3 interleaved repetitions on an AMD EPYC 9V74 (16C/32T, 32 MiB L3)
/// shared with other tenants:
///
/// `input` is the total over a graph's inputs; the `kvcat_*` rows take two large
/// past-KV tensors each, so the per-tensor size that actually decides the path
/// is half the figure shown.
///
/// | graph | input | serial | parallel | parallel/serial |
/// |---|--:|--:|--:|--:|
/// | `sm_decode_h32_kv1024` | 0.125 MiB | **4.8 us** | 108.5 | 22.6x worse |
/// | `sm_decode_h32_kv4096` | 0.5 MiB | **15.6** | 134.9 | 8.6x worse |
/// | `sm_decode_h32_kv8192` | 1 MiB | **31.5** | 130.4 | 4.1x worse |
/// | `rope_llama3_s1` | 2 MiB | **62.2** | 231.4 | 3.7x worse |
/// | `tr_bert_b8_s128` | 3 MiB | 372.6 | **120.9** | 3.1x better |
/// | `rope_llama3_s128` | 4 MiB | 353.4 | **255.9** | 1.4x better |
/// | `sm_bert_b8_s128` | 6 MiB | 391.8 | **126.1** | 3.1x better |
/// | `tr_llama3_s512` | 8 MiB | 589.8 | **178.3** | 3.3x better |
/// | `rope_llama3_s512` | 10 MiB | 718.0 | **314.9** | 2.3x better |
/// | `kvcat_llama3_p2047` | 16 MiB | 698.8 | **645.6** | 1.08x better |
/// | `kvcat_llama3_p4095` | 32 MiB | 1714.9 | **1525.6** | 1.12x better |
/// | `kvcat_llama3_p8191` | 64 MiB | 3501.3 | **3032.9** | 1.15x better |
/// | `kvcat_llama3_b8_p2047` | 128 MiB | 7377.6 | **7188.5** | 1.03x better |
///
/// Two honest caveats about that table:
///
/// * The cliff between 2 MiB (3.7x worse) and 3 MiB (3.1x better) is not really
///   a *size* effect. Below it the serial copy runs at 33-34 GB/s, which is a
///   cache-resident rate; above it, at 8-12 GB/s, which is a DRAM rate. What the
///   threshold actually separates is "the input survives in L3 between runs"
///   from "it does not", and byte count is only a proxy for that. A model whose
///   working set happens to stay resident above the threshold would pay the
///   fan-out for nothing.
/// * That fan-out is ~110 us here, which is large for Rayon and reflects a
///   contended host where waking eight sleeping workers waits on the OS
///   scheduler. On an idle machine it is smaller and the threshold could be
///   lower; it is not tuned for the idle case because production hosts are not
///   idle.
///
/// So the threshold sits at 4 MiB: above every measured loss, at or below every
/// measured win, and one whole step clear of the 2 MiB cliff rather than
/// balanced on it. The 3 MiB win is knowingly left on the floor to buy that
/// margin.
///
/// `ONNX_GENAI_HOST_COPY_PARALLEL_MIN_BYTES` overrides it, which is how the
/// ladder above was swept; `0` disables the parallel path entirely.
const HOST_COPY_PARALLEL_MIN_BYTES: usize = 4 << 20;

/// Parse an override string. Split out so it can be tested without touching
/// process-global state.
fn parse_host_copy_threshold(raw: Option<&str>) -> Option<usize> {
    raw?.trim().parse::<usize>().ok()
}

#[cfg(test)]
thread_local! {
    /// Per-thread threshold override for tests.
    ///
    /// Tests must not use the environment variable: `cargo test` runs a
    /// binary's tests on parallel threads, and `set_var` racing another
    /// thread's `getenv` is a data race that Rust 2024 explicitly forbids.
    /// A thread-local is race-free and needs no test serialisation.
    static TEST_HOST_COPY_THRESHOLD: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
}

/// The active threshold, honouring the environment override.
///
/// The environment is read exactly once per process and cached, so this is a
/// relaxed atomic load on every copy rather than a locked `getenv` plus a
/// `String` allocation. That matters: the small-copy path this function gates
/// is 4.8 us for a 0.125 MiB decode input, and an `env::var` miss is ~72 ns
/// against it - on the path the change is supposed to leave alone. Caching is
/// correct for the sweep that produced the table above because that sweep runs
/// one process per arm.
fn host_copy_parallel_min_bytes() -> usize {
    #[cfg(test)]
    if let Some(forced) = TEST_HOST_COPY_THRESHOLD.with(std::cell::Cell::get) {
        return forced;
    }
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        parse_host_copy_threshold(
            std::env::var("ONNX_GENAI_HOST_COPY_PARALLEL_MIN_BYTES")
                .ok()
                .as_deref(),
        )
        .unwrap_or(HOST_COPY_PARALLEL_MIN_BYTES)
    })
}

/// Workers available to a host copy: the decode budget, never the raw machine.
///
/// Only call this once a copy is known to be over the threshold:
/// `rayon::current_num_threads()` initialises the global Rayon pool, so calling
/// it on the small-copy path would spawn worker threads in processes that never
/// run a parallel kernel.
///
/// `rayon::current_num_threads()` is the whole machine unless something bounded
/// the global pool. `--native-threads` / `ONNX_GENAI_CPU_DECODE_THREADS` does
/// bound it (`bound_process_to_decode_budget`), so reading the pool here
/// respects an explicit budget while still capping at 8. The cap is an a priori
/// choice, not a measured optimum - a host copy is bandwidth-bound, so widening
/// the fan-out should only add workers to wake for the same DRAM - and the whole
/// ladder above was measured at this cap. No width sweep is included, so 8 is
/// not demonstrated to beat 16.
fn host_copy_workers() -> usize {
    rayon::current_num_threads().clamp(1, 8)
}

#[cfg(test)]
thread_local! {
    /// Test-only count of copies that actually took the parallel path, per
    /// thread.
    ///
    /// Without it a test that sets the threshold to 1 proves nothing on a
    /// machine whose Rayon pool is one worker wide - it would silently measure
    /// the serial path twice and pass. CI runners are exactly such machines. It
    /// is thread-local, not a global atomic, so a concurrently running test
    /// cannot bump another test's count and make its non-vacuity check pass
    /// spuriously.
    static PARALLEL_HOST_COPIES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// `dst.copy_from_slice(src)`, split across the decode budget when that is
/// faster and byte-for-byte identical either way.
///
/// May be called from inside a Rayon worker. The nested `par_chunks` then runs
/// on the same global pool, which Rayon supports - the outer worker
/// participates rather than blocking - so there is no deadlock, only a wider
/// split than the caller may expect.
fn copy_host_bytes(src: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(src.len(), dst.len());
    // Size first, and *only* size: this is a cached atomic load, whereas
    // `host_copy_workers` calls `rayon::current_num_threads()`, which
    // initialises the global Rayon pool on first use. Asking it here would
    // spawn a thread pool in every process that binds a graph input, including
    // ones that never run a parallel kernel. Below the threshold nothing may
    // touch Rayon.
    let threshold = host_copy_parallel_min_bytes();
    if threshold == 0 || src.len() < threshold {
        dst.copy_from_slice(src);
        return;
    }
    let workers = host_copy_workers();
    if workers < 2 {
        dst.copy_from_slice(src);
        return;
    }
    // Equal contiguous chunks, so no worker straddles the boundary and the
    // union is exactly `src`. `par_chunks_mut`/`par_chunks` walk the same
    // split, so chunk `i` of the destination receives chunk `i` of the source.
    #[cfg(test)]
    PARALLEL_HOST_COPIES.with(|n| n.set(n.get() + 1));
    let chunk = src.len().div_ceil(workers);
    use rayon::prelude::*;
    dst.par_chunks_mut(chunk)
        .zip(src.par_chunks(chunk))
        .for_each(|(d, s)| d.copy_from_slice(s));
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
        crate::kernels::matmul_nbits::try_bound_process_to_decode_budget()
            .map_err(EpError::KernelFailed)?;
        self.initialized = true;
        Ok(())
    }

    fn shutdown(&mut self) -> Result<()> {
        self.initialized = false;
        Ok(())
    }

    /// This EP never hands a node back to the host.
    ///
    /// # Why there is no performance-based decline
    ///
    /// An earlier design measured each op against ORT's CPU kernel and
    /// declined the shape/dtype ranges it lost, so ORT's CPU EP ran those
    /// nodes instead. That is withdrawn as an architectural rule: when this EP
    /// is selected it owns every node it supports, and a range where it is
    /// slower than ORT is a kernel to optimize, not a node to give away.
    ///
    /// The reasons are structural rather than about any single ratio. A
    /// deferral splits the graph, so the session pays a partition boundary and
    /// loses the fusion, prepack and buffer reuse that only hold inside one
    /// partition — and mixed ownership makes latency depend on which side won a
    /// microbenchmark on the machine the thresholds were tuned on. Selecting
    /// this EP is a request for this EP.
    ///
    /// There is no longer a routing-preference hook to override: the
    /// `ClaimPreference` type and the `claim_preference`/`claim_preference_node`
    /// trait methods were deleted from `onnx-runtime-ep-api`, so
    /// [`ExecutionProvider::supports_op`] is the only claim-time answer and a
    /// deferral cannot be reintroduced by overriding a default.
    ///
    /// The measured ORT gaps this EP still has to close are tracked in
    /// `docs/performance/CPU_ACTIVATION_GAPS.md`.
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
        if op.op_type == "DsaIndexSelect"
            && op.domain == "pkg.nxrt"
            && let Some(reason) =
                crate::kernels::dsa_index_select::unsupported_reason(op, shapes, input_dtypes)
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
        if op.op_type == "TensorScatter"
            && op.domain.is_empty()
            && let Some(reason) =
                crate::kernels::tensor_scatter::tensor_scatter_unsupported_reason(input_dtypes)
        {
            return KernelMatch::unsupported(reason);
        }
        if op.op_type == "STFT"
            && op.domain.is_empty()
            && let Some(reason) = crate::kernels::stft::unsupported_dtype_reason(input_dtypes)
        {
            return KernelMatch::unsupported(reason);
        }
        if op.op_type == "Einsum"
            && op.domain.is_empty()
            && let Some(reason) =
                crate::kernels::einsum::unsupported_reason(op, shapes, input_dtypes)
        {
            return KernelMatch::unsupported(reason);
        }
        // Attribute-level capability limits for the attention and MoE family.
        //
        // Every one of these mirrors a rejection its kernel factory raises, and
        // mirroring is mandatory rather than tidy: `GetCapability` consults only
        // the dtype and shape filters, so a factory rejection arrives *after*
        // ORT has compiled the node onto this EP and is a hard session failure
        // that no fallback recovers from. Claiming an op we cannot configure is
        // therefore strictly worse for the user than never claiming it — it
        // turns a model that runs on ORT's CPU EP into one that does not load.
        //
        // Each `unsupported_reason` runs the factory's own attribute parse, so
        // a new limit cannot be added to a factory without appearing here.
        //
        // These are *capability* declines, never performance ones: every entry
        // is a kernel we owe. See the benchmark doc's §23.6.
        let attribute_reason = match (op.op_type.as_str(), domain) {
            ("MoE", "com.microsoft") => crate::kernels::moe::unsupported_reason(op),
            ("QMoE", "com.microsoft") => crate::kernels::qmoe::unsupported_reason(op),
            ("GroupQueryAttention", "com.microsoft") => {
                crate::kernels::group_query_attention::unsupported_reason(op)
            }
            ("MultiHeadAttention", "com.microsoft") => {
                crate::kernels::multi_head_attention::unsupported_reason(op)
            }
            ("Attention", "com.microsoft") => {
                crate::kernels::msft_attention::unsupported_reason(op)
            }
            ("Attention", "ai.onnx") => crate::kernels::attention::unsupported_reason(op),
            _ => None,
        };
        if let Some(reason) = attribute_reason {
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
        let mut versioned = op.clone();
        if versioned.version.is_none() {
            versioned.version = i64::try_from(opset).ok();
        }
        factory.create(&versioned, shapes)
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

    /// Host upload, parallelised above a calibrated size.
    ///
    /// The session copies every graph input through here once per run, so for
    /// a large activation or KV-cache input this is the single biggest item in
    /// a run - measured at 3.6 ms of a 7.3 ms `Concat` benchmark, moving 64 MiB
    /// at 18.5 GB/s on one core. Splitting the copy across the decode budget
    /// reaches roughly 37 GB/s on this host.
    ///
    /// It is *only* a win once the copy is large enough to pay for the
    /// fan-out - below a few MiB the input is still cache-resident and one core
    /// already saturates it, so splitting costs up to 22x. Hence the threshold,
    /// and hence the base implementation stays exactly as it was below it. See
    /// [`HOST_COPY_PARALLEL_MIN_BYTES`] for the measured ladder.
    fn copy_from_host(&self, src: &[u8], dst: &mut DeviceBuffer) -> Result<()> {
        if !dst.device().is_host_accessible() {
            return Err(EpError::KernelFailed(format!(
                "{}: host upload is not implemented for device {:?}",
                self.name(),
                dst.device()
            )));
        }
        if src.len() > dst.len() {
            return Err(EpError::KernelFailed(format!(
                "{}: host upload of {} bytes exceeds destination {} bytes",
                self.name(),
                src.len(),
                dst.len()
            )));
        }
        if src.is_empty() {
            return Ok(());
        }
        // SAFETY: host accessibility and `src.len() <= dst.len()` are checked
        // above, and `dst` is uniquely borrowed, so this names a live host
        // allocation of at least `src.len()` bytes that nothing else aliases.
        //
        // The default implementation this overrides writes through the raw
        // pointer instead of forming a slice, which avoids naming
        // possibly-uninitialised memory as `&mut [u8]`. That distinction is
        // immaterial for `u8`: it has no invalid bit patterns and no niche, so
        // every byte of a fresh allocation is a valid `u8` whatever it holds,
        // and the reference is only written through.
        let out =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<u8>(), src.len()) };
        copy_host_bytes(src, out);
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
mod host_copy_tests {
    use super::{
        HOST_COPY_PARALLEL_MIN_BYTES, PARALLEL_HOST_COPIES, TEST_HOST_COPY_THRESHOLD,
        copy_host_bytes, host_copy_parallel_min_bytes, host_copy_workers,
        parse_host_copy_threshold,
    };
    use onnx_runtime_ep_api::{DeviceType, ExecutionProvider};

    fn ramp(len: usize) -> Vec<u8> {
        (0..len)
            .map(|i| (i.wrapping_mul(31) ^ (i >> 5)) as u8)
            .collect()
    }

    /// Force the threshold for this thread only, restoring it afterwards even
    /// on panic. Deliberately not the environment variable: `cargo test` runs
    /// these on parallel threads, and `set_var` racing another thread's
    /// `getenv` is a data race under Rust 2024 as well as a flake source.
    fn with_threshold<R>(bytes: usize, f: impl FnOnce() -> R) -> R {
        struct Restore(Option<usize>);
        impl Drop for Restore {
            fn drop(&mut self) {
                TEST_HOST_COPY_THRESHOLD.with(|c| c.set(self.0));
            }
        }
        let _restore = Restore(TEST_HOST_COPY_THRESHOLD.with(std::cell::Cell::get));
        TEST_HOST_COPY_THRESHOLD.with(|c| c.set(Some(bytes)));
        f()
    }

    fn parallel_copies_on_this_thread() -> usize {
        PARALLEL_HOST_COPIES.with(std::cell::Cell::get)
    }

    /// The whole point: whichever side of the threshold a copy lands on, the
    /// destination bytes are identical. A chunking bug that dropped, doubled or
    /// misordered a boundary chunk would show here.
    #[test]
    fn the_parallel_and_serial_copies_produce_identical_bytes() {
        // Sizes that straddle the chunk split awkwardly: primes, one byte over
        // a chunk boundary, and a size that is not a multiple of the worker
        // count.
        for &len in &[
            1usize,
            7,
            4095,
            4096,
            4097,
            1 << 20,
            (1 << 20) + 13,
            3 * (1 << 20) + 1,
        ] {
            let src = ramp(len);
            let mut serial = vec![0u8; len];
            let mut parallel = vec![0u8; len];

            // Both paths are exercised explicitly by moving the threshold, not
            // by hoping `len` lands on either side of it.
            with_threshold(0, || copy_host_bytes(&src, &mut serial));
            let before = parallel_copies_on_this_thread();
            with_threshold(1, || copy_host_bytes(&src, &mut parallel));
            let took_parallel = parallel_copies_on_this_thread() > before;

            // On a single-worker pool the "parallel" arm is the serial path and
            // proves nothing; say so rather than passing vacuously.
            if host_copy_workers() >= 2 {
                assert!(
                    took_parallel,
                    "threshold 1 did not reach the parallel path at len {len}"
                );
            }
            assert_eq!(serial, src, "serial copy is not the source at len {len}");
            assert_eq!(
                parallel, src,
                "parallel copy diverges from the source at len {len}"
            );
        }
    }

    /// A falsifier for the calibration, not a restatement of the constant. The
    /// in-session ladder (see the constant's docs) measured 2 MiB inputs 3.7x
    /// *slower* with threads and 4 MiB inputs 1.4x faster, so the threshold has
    /// to land strictly between them. Retuning inside that band is fine;
    /// moving outside it has to fail the build.
    ///
    /// This is a compile-time check on purpose: a mis-tuned constant should
    /// never get as far as a test run.
    const _CALIBRATION_BAND: () = {
        assert!(
            HOST_COPY_PARALLEL_MIN_BYTES > 2 << 20,
            "2 MiB inputs were 3.7x slower parallel in-session; the threshold must exclude them"
        );
        assert!(
            HOST_COPY_PARALLEL_MIN_BYTES <= 4 << 20,
            "4 MiB inputs were 1.4x faster parallel in-session; the threshold must admit them"
        );
    };

    /// The override parser has to accept what the sweep passes it and reject
    /// what it does not, or the ladder that produced the table above measured
    /// nothing. Tested on the parser rather than through the environment so it
    /// cannot race a concurrently running test.
    #[test]
    fn the_override_parser_accepts_what_the_sweep_passes_it() {
        assert_eq!(parse_host_copy_threshold(Some("12345")), Some(12345));
        assert_eq!(parse_host_copy_threshold(Some("  0  ")), Some(0));
        assert_eq!(parse_host_copy_threshold(None), None);
        assert_eq!(parse_host_copy_threshold(Some("")), None);
        assert_eq!(parse_host_copy_threshold(Some("-1")), None);
        assert_eq!(parse_host_copy_threshold(Some("4MiB")), None);
    }

    /// A forced threshold has to actually reach `copy_host_bytes`, or every
    /// other test here is measuring the default and proving nothing.
    #[test]
    fn a_forced_threshold_replaces_the_calibrated_one() {
        with_threshold(12345, || {
            assert_eq!(host_copy_parallel_min_bytes(), 12345);
        });
        if std::env::var_os("ONNX_GENAI_HOST_COPY_PARALLEL_MIN_BYTES").is_none() {
            assert_eq!(host_copy_parallel_min_bytes(), HOST_COPY_PARALLEL_MIN_BYTES);
        }
    }

    /// The fan-out is capped at the width the ladder was measured with,
    /// regardless of how wide the Rayon pool is.
    #[test]
    fn the_worker_count_is_capped_at_the_measured_optimum() {
        let workers = host_copy_workers();
        assert!(
            (1..=8).contains(&workers),
            "host copy fan-out {workers} left the measured 1..=8 range"
        );
    }

    /// A below-threshold copy must decide using size alone.
    ///
    /// `host_copy_workers` calls `rayon::current_num_threads()`, which
    /// initialises the global Rayon pool. If the worker query ran before the
    /// size gate, every process that binds a graph input would spawn a thread
    /// pool - including ones that never run a parallel kernel. CI caught
    /// exactly that: a session test binary that had never used Rayon started
    /// one, and Miri then reported crossbeam-epoch's teardown as UB.
    ///
    /// This asserts only the observable half of that property: with the
    /// threshold at its calibrated value, a small copy is correct and stays
    /// serial. It is named for what it observes, not for the ordering - it
    /// would still pass with the gates swapped back, because the size gate
    /// returns first either way. The
    /// ordering itself is enforced by the Miri job, which is the real falsifier
    /// and fails if the query moves back above the gate - `cargo test` gives no
    /// ordering guarantee, so a process-global "was the pool started" assertion
    /// would be answered by whichever other test ran first.
    #[test]
    fn a_small_copy_stays_serial() {
        let src = ramp(4096);
        let mut dst = vec![0u8; 4096];
        with_threshold(HOST_COPY_PARALLEL_MIN_BYTES, || {
            let before = parallel_copies_on_this_thread();
            copy_host_bytes(&src, &mut dst);
            assert_eq!(
                parallel_copies_on_this_thread(),
                before,
                "a 4 KiB copy took the parallel path"
            );
        });
        assert_eq!(dst, src);
    }

    /// End-to-end through the EP: a real upload larger than the threshold must
    /// land every byte in the destination buffer.
    #[test]
    fn a_large_upload_through_the_provider_lands_every_byte() {
        let ep = super::CpuExecutionProvider::new();
        assert_eq!(ep.device_type(), DeviceType::Cpu);
        let len = (HOST_COPY_PARALLEL_MIN_BYTES) + 4097;
        let src = ramp(len);
        let mut buf = ep.allocate(len, 64).expect("allocate");
        ep.copy_from_host(&src, &mut buf).expect("upload");
        // SAFETY: `buf` is a live host allocation of `len` bytes from this EP.
        let got = unsafe { std::slice::from_raw_parts(buf.as_ptr().cast::<u8>(), len) };
        assert_eq!(got, &src[..]);
        ep.deallocate(buf).expect("free");
    }

    /// An upload smaller than the destination must not touch the tail, and one
    /// larger than the destination must be rejected rather than overrun it.
    #[test]
    fn a_short_upload_leaves_the_tail_alone_and_an_oversized_one_is_rejected() {
        let ep = super::CpuExecutionProvider::new();
        let cap = 1 << 12;
        let mut buf = ep.allocate(cap, 64).expect("allocate");
        // SAFETY: `buf` is a live host allocation of `cap` bytes.
        unsafe { std::ptr::write_bytes(buf.as_mut_ptr().cast::<u8>(), 0xAB, cap) };
        let src = ramp(cap / 2);
        ep.copy_from_host(&src, &mut buf).expect("short upload");
        // SAFETY: as above.
        let got = unsafe { std::slice::from_raw_parts(buf.as_ptr().cast::<u8>(), cap) };
        assert_eq!(&got[..cap / 2], &src[..]);
        assert!(
            got[cap / 2..].iter().all(|&b| b == 0xAB),
            "a short upload overwrote bytes past its own length"
        );
        assert!(ep.copy_from_host(&ramp(cap + 1), &mut buf).is_err());
        ep.deallocate(buf).expect("free");
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
            let shapes = if *op == "Einsum" {
                node.attributes
                    .insert("equation".into(), Attribute::String(b"i->i".to_vec()));
                vec![vec![2]]
            } else {
                Vec::new()
            };
            if *op == "BitShift" {
                node.attributes
                    .insert("direction".into(), Attribute::String(b"RIGHT".to_vec()));
            }
            assert!(
                ep.get_kernel(&node, &shapes, 18).is_ok(),
                "no kernel for {op}"
            );
        }
        let bad = Node::new(onnx_runtime_ir::NodeId(99), "UnknownOp", vec![], vec![]);
        assert!(ep.get_kernel(&bad, &[], 17).is_err());
    }

    /// Every attribute a kernel factory rejects must also be rejected by
    /// `supports_op`, because a factory rejection lands after ORT has compiled
    /// the node onto this EP and is a hard session failure no fallback recovers
    /// from — strictly worse than never claiming. This is the pure-Rust half of
    /// `plugin_ort_e2e::factory_only_capability_limits_are_declined_at_claim_time`,
    /// so a regression is caught even where ORT is unavailable.
    #[test]
    fn every_factory_attribute_rejection_is_mirrored_at_claim_time() {
        let ep = CpuExecutionProvider::new();
        /// `(op, domain, opset, rejecting attribute, prerequisite attributes)`.
        type FactoryRejection = (
            &'static str,
            &'static str,
            u64,
            (&'static str, Attribute),
            &'static [(&'static str, Attribute)],
        );
        let cases: &[FactoryRejection] = &[
            (
                "MoE",
                "com.microsoft",
                1,
                ("use_sparse_mixer", Attribute::Int(1)),
                &[],
            ),
            (
                "QMoE",
                "com.microsoft",
                1,
                ("block_size", Attribute::Int(0)),
                &[],
            ),
            (
                "QMoE",
                "com.microsoft",
                1,
                ("expert_weight_bits", Attribute::Int(3)),
                &[("block_size", Attribute::Int(32))],
            ),
            (
                "GroupQueryAttention",
                "com.microsoft",
                1,
                ("smooth_softmax", Attribute::Int(1)),
                &[
                    ("num_heads", Attribute::Int(4)),
                    ("kv_num_heads", Attribute::Int(2)),
                ],
            ),
            (
                "GroupQueryAttention",
                "com.microsoft",
                1,
                ("qk_output", Attribute::Int(1)),
                &[
                    ("num_heads", Attribute::Int(4)),
                    ("kv_num_heads", Attribute::Int(2)),
                ],
            ),
            (
                "GroupQueryAttention",
                "com.microsoft",
                1,
                ("kv_cache_bit_width", Attribute::Int(8)),
                &[
                    ("num_heads", Attribute::Int(4)),
                    ("kv_num_heads", Attribute::Int(2)),
                ],
            ),
            (
                "Attention",
                "com.microsoft",
                1,
                ("do_rotary", Attribute::Int(1)),
                &[("num_heads", Attribute::Int(4))],
            ),
            (
                "Attention",
                "com.microsoft",
                1,
                ("past_present_share_buffer", Attribute::Int(1)),
                &[("num_heads", Attribute::Int(4))],
            ),
            (
                "Attention",
                "ai.onnx",
                23,
                ("qk_matmul_output_mode", Attribute::Int(7)),
                &[],
            ),
            (
                "MultiHeadAttention",
                "com.microsoft",
                1,
                ("num_heads", Attribute::Int(0)),
                &[],
            ),
        ];

        for (op, domain, opset, (attr, value), extra) in cases {
            let mut node = Node::new(NodeId(0), *op, vec![], vec![]);
            node.domain = (*domain).into();
            for (name, extra_value) in *extra {
                node.attributes.insert((*name).into(), extra_value.clone());
            }
            node.attributes.insert((*attr).into(), value.clone());

            let factory = ep.get_kernel(&node, &[], *opset);
            assert!(
                factory.is_err(),
                "{domain}::{op} with {attr} is expected to be a factory rejection; if the \
                 kernel now implements it, drop this row rather than the claim-time guard"
            );

            let claim = ep.supports_op(&node, *opset, &[], &[], &[]);
            assert!(
                !claim.is_supported(),
                "{domain}::{op} with {attr} is claimed but its factory rejects it, so the \
                 session dies at CreateSession with no fallback. Mirror the limit in \
                 `supports_op`."
            );
        }

        // `PackedMultiHeadAttention` needs a fully-formed node rather than a
        // table row. Its claim-time check inspects inputs, shapes and dtypes
        // before it reaches the attribute mirror, so an attribute-only node
        // declines for the *wrong* reason and the assertion would pass even
        // with the mirror deleted — which is exactly what review demonstrated.
        // Hence the explicit reason match.
        let mut graph = Graph::new();
        let value = |graph: &mut Graph, name: &str, dtype| {
            Some(graph.create_named_value(name.to_string(), dtype, static_shape([])))
        };
        let inputs = vec![
            value(&mut graph, "query", DataType::Float32),
            value(&mut graph, "key", DataType::Float32),
            value(&mut graph, "value", DataType::Float32),
            None,
            value(&mut graph, "token_offset", DataType::Int32),
            value(&mut graph, "cumulative_sequence_length", DataType::Int32),
        ];
        let output =
            graph.create_named_value("output".to_string(), DataType::Float32, static_shape([]));
        let mut node = Node::new(NodeId(0), "PackedMultiHeadAttention", inputs, vec![output]);
        node.domain = "com.microsoft".into();
        node.attributes
            .insert("num_heads".into(), Attribute::Int(2));
        node.attributes
            .insert("scale".into(), Attribute::Float(f32::INFINITY));
        let shapes = [
            static_shape([4, 8]),
            static_shape([4, 8]),
            static_shape([4, 8]),
            static_shape([]),
            static_shape([1, 4]),
            static_shape([2]),
        ];
        let dtypes = [
            DataType::Float32,
            DataType::Float32,
            DataType::Float32,
            DataType::Float32,
            DataType::Int32,
            DataType::Int32,
        ];
        assert!(
            ep.get_kernel(&node, &[], 1).is_err(),
            "PackedMultiHeadAttention with a non-finite scale is expected to be a factory \
             rejection"
        );
        let claim = ep.supports_op(&node, 1, &shapes, &dtypes, &[]);
        let reason = claim.reason().unwrap_or_default().to_string();
        assert!(
            reason.contains("scale must be finite"),
            "PackedMultiHeadAttention's attribute mirror was not reached; the node declined \
             for `{reason}` instead, so this case would not catch the mirror being removed"
        );
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
    fn stft_claims_only_the_kernel_compute_dtypes() {
        let ep = CpuExecutionProvider::new();
        let stft = Node::new(onnx_runtime_ir::NodeId(0), "STFT", vec![], vec![]);
        for dtype in [DataType::Float32, DataType::Float16, DataType::BFloat16] {
            assert!(
                ep.supports_op(
                    &stft,
                    17,
                    &[],
                    &[dtype, DataType::Int64, DataType::Undefined, DataType::Int64],
                    &[],
                )
                .is_supported(),
                "{dtype:?} STFT should be claimed"
            );
        }

        let rejected = ep.supports_op(
            &stft,
            17,
            &[],
            &[
                DataType::Float64,
                DataType::Int64,
                DataType::Undefined,
                DataType::Int64,
            ],
            &[],
        );
        assert!(!rejected.is_supported());
        assert!(
            rejected.reason().unwrap().contains("computes in f32"),
            "{:?}",
            rejected.reason()
        );
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

    /// Builds a `pkg.nxrt::DsaIndexSelect` node plus the `(shapes, dtypes)`
    /// claim metadata for a compatible baseline: batch=1, q_seq=2, heads=2,
    /// head_dim=4, key_seq=3, top_k=2. Each shape/dtype axis matches
    /// `DsaIndexSelectKernel::derive_dims`'s cross-input contract exactly, so
    /// callers can flip exactly one axis to prove exactly one rejection.
    fn dsa_index_select_baseline() -> (Node, Vec<Shape>, Vec<DataType>) {
        let mut graph = Graph::new();
        let value = |graph: &mut Graph, name: &str, dtype| {
            Some(graph.create_named_value(name.to_string(), dtype, static_shape([])))
        };
        let inputs = vec![
            value(&mut graph, "query", DataType::Float32),
            value(&mut graph, "key", DataType::Float32),
            value(&mut graph, "weights", DataType::Float32),
            value(&mut graph, "attention_bias", DataType::Float32),
        ];
        let output = graph.create_named_value(
            "selected_indices".to_string(),
            DataType::Int64,
            static_shape([]),
        );
        let mut node = Node::new(NodeId(0), "DsaIndexSelect", inputs, vec![output]);
        node.domain = "pkg.nxrt".into();
        node.attributes.insert("top_k".into(), Attribute::Int(2));
        node.attributes
            .insert("scale".into(), Attribute::Float(1.0));
        let shapes = vec![
            static_shape([1, 2, 2, 4]), // query (B, S, H, D)
            static_shape([1, 3, 4]),    // key (B, T, D)
            static_shape([1, 2, 2]),    // weights (B, S, H)
            static_shape([1, 1, 2, 3]), // attention_bias (B, 1, S, T)
        ];
        let dtypes = vec![
            DataType::Float32,
            DataType::Float32,
            DataType::Float32,
            DataType::Float32,
        ];
        (node, shapes, dtypes)
    }

    #[test]
    fn dsa_index_select_claim_gate_accepts_a_compatible_node() {
        let ep = CpuExecutionProvider::new();
        let (node, shapes, dtypes) = dsa_index_select_baseline();
        let claim = ep.supports_op(&node, 1, &shapes, &dtypes, &[]);
        assert!(
            claim.is_supported(),
            "a shape/dtype/attribute-compatible DsaIndexSelect node must be claimed; got {:?}",
            claim.reason()
        );
        // Claim/execute mirroring: the factory must accept the same node
        // (`DsaIndexSelectFactory::create` only parses attributes, so an
        // empty shape list is the correct call shape here).
        assert!(
            ep.get_kernel(&node, &[], 1).is_ok(),
            "a node the claim gate accepts must also be instantiable by the factory"
        );
    }

    #[test]
    fn dsa_index_select_claim_gate_rejects_non_f32_attention_bias() {
        // `attention_bias` carries the `-inf`/finfo.min causal-fill sentinel,
        // which is only meaningful at Float32 precision; the kernel and the
        // claim gate both hard-require Float32 for input 3 regardless of the
        // query/key/weights element type.
        let ep = CpuExecutionProvider::new();
        let (node, shapes, mut dtypes) = dsa_index_select_baseline();
        dtypes[3] = DataType::Float16;
        let rejected = ep.supports_op(&node, 1, &shapes, &dtypes, &[]);
        assert!(!rejected.is_supported());
        let reason = rejected
            .reason()
            .expect("DsaIndexSelect claim must be denied");
        assert!(
            reason.contains("attention_bias") && reason.contains("expected Float32"),
            "{reason}"
        );
        // Mirror: the factory must reject with the same reasoning family, not
        // silently accept a node the claim gate declined.
        let mut execute_dtypes = dtypes.clone();
        execute_dtypes[3] = DataType::Float16;
        assert!(
            crate::kernels::dsa_index_select::unsupported_reason(&node, &shapes, &execute_dtypes)
                .unwrap()
                .contains("expected Float32")
        );
    }

    #[test]
    fn dsa_index_select_claim_gate_rejects_incompatible_shape() {
        // query/key head_dim mismatch (query D=4, key D=5): a cross-input
        // shape contract violation, not a rank or dtype issue.
        let ep = CpuExecutionProvider::new();
        let (node, mut shapes, dtypes) = dsa_index_select_baseline();
        shapes[1] = static_shape([1, 3, 5]);
        let rejected = ep.supports_op(&node, 1, &shapes, &dtypes, &[]);
        assert!(!rejected.is_supported());
        let reason = rejected
            .reason()
            .expect("DsaIndexSelect claim must be denied");
        assert!(reason.contains("query/key head_dim"), "{reason}");
    }

    #[test]
    fn dsa_index_select_claim_gate_rejects_bad_bias_head_broadcast() {
        // `attention_bias` dim 1 must be exactly 1 (head-broadcast); a wider
        // dim 1 is a distinct rejection axis from rank or cross-seq mismatches.
        let ep = CpuExecutionProvider::new();
        let (node, mut shapes, dtypes) = dsa_index_select_baseline();
        shapes[3] = static_shape([1, 2, 2, 3]);
        let rejected = ep.supports_op(&node, 1, &shapes, &dtypes, &[]);
        assert!(!rejected.is_supported());
        let reason = rejected
            .reason()
            .expect("DsaIndexSelect claim must be denied");
        assert!(reason.contains("head-broadcast"), "{reason}");
    }

    #[test]
    fn dsa_index_select_claim_gate_rejects_non_positive_top_k() {
        let ep = CpuExecutionProvider::new();
        let (mut node, shapes, dtypes) = dsa_index_select_baseline();
        node.attributes.insert("top_k".into(), Attribute::Int(0));
        let rejected = ep.supports_op(&node, 1, &shapes, &dtypes, &[]);
        assert!(!rejected.is_supported());
        let reason = rejected
            .reason()
            .expect("DsaIndexSelect claim must be denied");
        assert!(
            reason.contains("top_k") && reason.contains("must be > 0"),
            "{reason}"
        );
        // Mirror: the factory (execute-time attribute parse) rejects for the
        // same reason, proving the claim gate did not invent a check the
        // kernel does not also enforce.
        assert!(ep.get_kernel(&node, &[], 1).is_err());
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
