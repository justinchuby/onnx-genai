//! Thin FFI wrapper around a vendored subset of ONNX Runtime's MLAS
//! single-precision GEMM (`MlasGemmBatch`).
//!
//! The vendored MLAS is compiled in its standalone `BUILD_MLAS_NO_ONNXRUNTIME`
//! mode, whose threading primitives normally serialize. This crate installs a
//! persistent work-stealing parallel-for backend (see [`ensure_threading`] and
//! `vendor/shim.cpp`) so MLAS keeps its own cache-aware GEMM tile partitioning
//! while executing the tiles across a low-overhead pool that is created once and
//! reused. QNBit callers can pass `multithread=true` to run one full-width
//! `MlasQNBitGemmBatch` call and let MLAS partition N internally, matching ORT's
//! intra-op threadpool shape. See `docs/performance/MLAS_SYS_SPIKE.md` for the original
//! single-thread feasibility spike.

use std::os::raw::c_int;
use std::os::raw::c_void;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Once, OnceLock};

mod work_stealing_pool;
pub use work_stealing_pool::WorkStealingThreadPool;

unsafe extern "C" {
    /// Vendored-MLAS SGEMM shim (single-threaded). Computes
    /// `C := alpha * op(A) * op(B) + beta * C` with row-major matrices.
    fn mlas_sgemm(
        trans_a: c_int,
        trans_b: c_int,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: *const f32,
        lda: usize,
        b: *const f32,
        ldb: usize,
        beta: f32,
        c: *mut f32,
        ldc: usize,
    );

    #[allow(clippy::too_many_arguments)]
    fn mlas_sgemm_batch(
        trans_a: c_int,
        trans_b: c_int,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: *const *const f32,
        lda: usize,
        b: *const *const f32,
        ldb: usize,
        beta: f32,
        c: *const *mut f32,
        ldc: usize,
        batch_size: usize,
    );

    #[allow(clippy::too_many_arguments)]
    fn mlas_qgemm_i32(
        m: usize,
        n: usize,
        k: usize,
        a_is_signed: c_int,
        b_is_signed: c_int,
        a: *const c_void,
        lda: usize,
        zero_point_a: u8,
        b: *const c_void,
        ldb: usize,
        zero_point_b: *const u8,
        per_column_zero_points: c_int,
        c: *mut i32,
        ldc: usize,
        multithread: c_int,
    );

    fn mlas_qgemm_pack_b_size(n: usize, k: usize, a_is_signed: c_int, b_is_signed: c_int) -> usize;
    fn mlas_qgemm_pack_b(
        n: usize,
        k: usize,
        b: *const c_void,
        ldb: usize,
        a_is_signed: c_int,
        b_is_signed: c_int,
        packed_b: *mut c_void,
    );
    fn mlas_qgemm_i32_packed(
        m: usize,
        n: usize,
        k: usize,
        a_is_signed: c_int,
        b_is_signed: c_int,
        a: *const c_void,
        lda: usize,
        zero_point_a: u8,
        packed_b: *const c_void,
        zero_point_b: *const u8,
        per_column_zero_points: c_int,
        c: *mut i32,
        ldc: usize,
        multithread: c_int,
    );

    #[allow(clippy::too_many_arguments)]
    fn mlas_qgemm_requantize(
        m: usize,
        n: usize,
        k: usize,
        a_is_signed: c_int,
        b_is_signed: c_int,
        a: *const c_void,
        lda: usize,
        zero_point_a: u8,
        b: *const c_void,
        ldb: usize,
        b_is_packed: c_int,
        zero_point_b: *const u8,
        per_column_zero_points: c_int,
        c: *mut i32,
        ldc: usize,
        output: *mut c_void,
        output_ld: usize,
        output_is_signed: c_int,
        scale: *const f32,
        per_column_scale: c_int,
        output_zero_point: i32,
        multithread: c_int,
    );

    fn mlas_sgemm_pack_b_size(trans_a: c_int, trans_b: c_int, n: usize, k: usize) -> usize;
    fn mlas_sgemm_pack_b(
        trans_a: c_int,
        trans_b: c_int,
        n: usize,
        k: usize,
        b: *const f32,
        ldb: usize,
        packed_b: *mut u8,
    );
    fn mlas_sgemm_packed(
        trans_a: c_int,
        trans_b: c_int,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: *const f32,
        lda: usize,
        packed_b: *const u8,
        beta: f32,
        c: *mut f32,
        ldc: usize,
    );

    fn mlas_float_kernel_id() -> c_int;

    /// Vectorized logistic (sigmoid) over `n` contiguous f32s: single-threaded
    /// MLAS SIMD sigmoid, used to build SiLU without a scalar `expf` loop.
    fn mlas_compute_logistic(input: *const f32, output: *mut f32, n: usize);
    /// Vectorized fused SiLU over `n` contiguous f32s. MLAS runtime-dispatches
    /// to its one-pass AVX-512F kernel when supported.
    fn mlas_compute_silu(input: *const f32, output: *mut f32, n: usize);
    /// Vectorized `tanh` over `n` contiguous f32s — the same polynomial ONNX
    /// Runtime's own `Tanh` CPU kernel calls.
    fn mlas_compute_tanh(input: *const f32, output: *mut f32, n: usize);
    /// Vectorized `erf` over `n` contiguous f32s — the same polynomial ONNX
    /// Runtime's own `Erf` CPU kernel calls.
    fn mlas_compute_erf(input: *const f32, output: *mut f32, n: usize);
    /// Vectorized exact (erf-based) GELU over `n` contiguous f32s. Input and
    /// output must not overlap.
    fn mlas_compute_gelu_erf(input: *const f32, output: *mut f32, n: usize);
    /// Row-wise softmax over `n` rows of `d` contiguous f32s, single-threaded,
    /// using MLAS's SIMD max reduction and polynomial exp.
    fn mlas_compute_softmax_in_place(data: *mut f32, n: usize, d: usize);
    fn mlas_eltwise_add(left: *const f32, right: *const f32, output: *mut f32, n: usize);
    fn mlas_compute_activation(
        kind: c_int,
        minimum: f32,
        maximum: f32,
        input: *const f32,
        output: *mut f32,
        n: usize,
    );

    fn mlas_conv_prepare(
        dimensions: usize,
        batch_count: usize,
        group_count: usize,
        input_channels_per_group: usize,
        input_shape: *const i64,
        kernel_shape: *const i64,
        dilation_shape: *const i64,
        padding: *const i64,
        stride_shape: *const i64,
        output_shape: *const i64,
        filter_count_per_group: usize,
        working_buffer_elements: *mut usize,
    ) -> *mut c_void;
    fn mlas_conv_run(
        plan: *const c_void,
        input: *const f32,
        filter: *const f32,
        bias: *const f32,
        working_buffer: *mut f32,
        output: *mut f32,
    );
    fn mlas_conv_plan_destroy(plan: *mut c_void);

    // ---- NCHWc blocked convolution ----
    fn mlas_nchwc_block_size() -> usize;
    fn mlas_nchwc_reorder_input_nchw(
        source: *const f32,
        dest: *mut f32,
        channels: usize,
        input_size: usize,
    );
    fn mlas_nchwc_reorder_output_nchw(output_shape: *const i64, source: *const f32, dest: *mut f32);
    fn mlas_nchwc_reorder_filter_bibo(filter_shape: *const i64, source: *const f32, dest: *mut f32);
    fn mlas_nchwc_reorder_filter_bo(filter_shape: *const i64, source: *const f32, dest: *mut f32);
    #[allow(clippy::too_many_arguments)]
    fn mlas_nchwc_conv(
        input_shape: *const i64,
        kernel_shape: *const i64,
        dilation_shape: *const i64,
        padding: *const i64,
        stride_shape: *const i64,
        output_shape: *const i64,
        group_count: usize,
        input: *const f32,
        filter: *const f32,
        bias: *const f32,
        output: *mut f32,
        activation_kind: c_int,
        activation_value0: f32,
        activation_value1: f32,
        zero_mode: c_int,
    );
    fn mlas_pool(
        kind: c_int,
        dimensions: usize,
        input_shape: *const i64,
        kernel_shape: *const i64,
        padding: *const i64,
        stride_shape: *const i64,
        output_shape: *const i64,
        input: *const f32,
        output: *mut f32,
    );
    #[allow(clippy::too_many_arguments)]
    fn mlas_nchwc_pool(
        kind: c_int,
        input_shape: *const i64,
        kernel_shape: *const i64,
        dilation_shape: *const i64,
        padding: *const i64,
        stride_shape: *const i64,
        output_shape: *const i64,
        input: *const f32,
        output: *mut f32,
    );

    // ---- Blocked n-bit quantized GEMM (SQNBitGemm) ----
    fn mlas_qnbit_gemm_available(bits: usize, blk_len: usize, comp_type: c_int) -> c_int;
    fn mlas_qnbit_gemm_pack_b_size(
        n: usize,
        k: usize,
        bits: usize,
        blk_len: usize,
        has_zp: c_int,
        comp_type: c_int,
    ) -> usize;
    fn mlas_qnbit_gemm_pack_b(
        n: usize,
        k: usize,
        bits: usize,
        blk_len: usize,
        comp_type: c_int,
        quant_b_data: *const c_void,
        packed_b: *mut u8,
        quant_b_scale: *const f32,
        has_zp: c_int,
        quant_b_zero_point: *const c_void,
    );
    fn mlas_qnbit_gemm_workspace_size(
        m: usize,
        n: usize,
        k: usize,
        bits: usize,
        blk_len: usize,
        has_zp: c_int,
        comp_type: c_int,
    ) -> usize;
    #[allow(clippy::too_many_arguments)]
    fn mlas_qnbit_gemm(
        m: usize,
        n: usize,
        k: usize,
        bits: usize,
        blk_len: usize,
        comp_type: c_int,
        a: *const f32,
        lda: usize,
        packed_b: *const u8,
        quant_b_scale: *const f32,
        has_zp: c_int,
        quant_b_zero_point: *const c_void,
        bias: *const f32,
        c: *mut f32,
        ldc: usize,
        workspace: *mut u8,
        multithread: c_int,
    );

    /// Register the Rust-backed threading backend with the vendored MLAS
    /// standalone build (see `vendor/shim.cpp`). Passing the callbacks below
    /// lets MLAS's own GEMM tile partitioning run across a real thread pool.
    fn mlas_set_threading(
        parallel_for: MlasParallelForFn,
        max_threads: MlasMaxThreadsFn,
        rust_ctx: *mut c_void,
    );
}

/// One MLAS work unit: run partition `tid`. `task_ctx` is opaque C++ state.
type MlasTaskFn = unsafe extern "C" fn(task_ctx: *mut c_void, tid: isize);
/// Backend that runs `task(task_ctx, tid)` for every `tid` in `[0, iterations)`.
type MlasParallelForFn = unsafe extern "C" fn(
    rust_ctx: *mut c_void,
    iterations: isize,
    task: MlasTaskFn,
    task_ctx: *mut c_void,
);
/// Backend that reports the degree of parallelism MLAS may use.
type MlasMaxThreadsFn = unsafe extern "C" fn(rust_ctx: *mut c_void) -> c_int;

const MLAS_WORK_STEALING_THREADS_ENV: &str = "ONNX_GENAI_MLAS_THREADPOOL_THREADS";
/// The CPU EP's thread-budget knob. `mlas-sys` sits *below* the EP and cannot
/// depend on it, so the name lives here and the EP refers to *this* constant;
/// `onnx-runtime-ep-cpu` static-asserts that its own `DECODE_THREADS_ENV`
/// matches, so the two can never drift.
///
/// Without this, an embedder that asked the CPU EP for N threads still got a
/// standalone-MLAS pool sized by [`default_mlas_thread_count`]'s own default,
/// so the request was silently ignored for every `MlasGemmBatch` call.
pub const CPU_DECODE_THREADS_ENV: &str = "ONNX_GENAI_CPU_DECODE_THREADS";
/// Upper bound on the standalone pool, matching the previous `n.min(64)` cap on
/// the explicit override.
///
/// A budget above this is clamped, which makes the standalone pool smaller than
/// the CPU EP's persistent pool (that one clamps only to `available`). The
/// divergence is reported once on stderr rather than applied silently.
const MAX_MLAS_POOL_THREADS: usize = 64;

/// Programmatic thread budget, mirroring the CPU EP's
/// `set_decode_thread_budget`. Zero means unset.
static POOL_THREAD_BUDGET: AtomicUsize = AtomicUsize::new(0);

static MLAS_PARALLEL_FOR_CALLS: AtomicUsize = AtomicUsize::new(0);
static MLAS_PARALLEL_FOR_ITERATIONS: AtomicUsize = AtomicUsize::new(0);
static MLAS_PARALLEL_FOR_FALLBACKS: AtomicUsize = AtomicUsize::new(0);
static MLAS_PARALLEL_FOR_BLOCK_CALLS: AtomicUsize = AtomicUsize::new(0);
static MLAS_PARALLEL_FOR_BLOCKS_CLAIMED: AtomicUsize = AtomicUsize::new(0);

/// Snapshot of the vendored-MLAS standalone threading backend.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MlasThreadingStats {
    /// Number of `MlasStandaloneParallelFor` calls routed through Rust.
    pub parallel_for_calls: usize,
    /// Sum of MLAS partition indices scheduled through the backend.
    pub scheduled_iterations: usize,
    /// Number of calls that fell back to a serial loop because pool creation failed.
    pub serial_fallback_calls: usize,
    /// Degree of parallelism reported to MLAS by `MlasGetMaximumThreadCount`.
    pub pool_threads: usize,
    /// Number of MLAS callbacks run through ORT-style dynamic block claiming.
    pub dynamic_block_calls: usize,
    /// Number of individual MLAS work-item blocks claimed dynamically.
    pub dynamic_blocks_claimed: usize,
}

/// Return the current MLAS backend stats. Intended for diagnostics and microbenchmarks.
pub fn mlas_threading_stats() -> MlasThreadingStats {
    MlasThreadingStats {
        parallel_for_calls: MLAS_PARALLEL_FOR_CALLS.load(Ordering::Relaxed),
        scheduled_iterations: MLAS_PARALLEL_FOR_ITERATIONS.load(Ordering::Relaxed),
        serial_fallback_calls: MLAS_PARALLEL_FOR_FALLBACKS.load(Ordering::Relaxed),
        pool_threads: mlas_threading_degree(),
        dynamic_block_calls: MLAS_PARALLEL_FOR_BLOCK_CALLS.load(Ordering::Relaxed),
        dynamic_blocks_claimed: MLAS_PARALLEL_FOR_BLOCKS_CLAIMED.load(Ordering::Relaxed),
    }
}

/// Reset the MLAS backend stats. Intended for tests and microbenchmarks.
pub fn reset_mlas_threading_stats() {
    MLAS_PARALLEL_FOR_CALLS.store(0, Ordering::Relaxed);
    MLAS_PARALLEL_FOR_ITERATIONS.store(0, Ordering::Relaxed);
    MLAS_PARALLEL_FOR_FALLBACKS.store(0, Ordering::Relaxed);
    MLAS_PARALLEL_FOR_BLOCK_CALLS.store(0, Ordering::Relaxed);
    MLAS_PARALLEL_FOR_BLOCKS_CLAIMED.store(0, Ordering::Relaxed);
}

/// MLAS `MlasQNBitGemmBatch` thread partitioning for a single call.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SQNBitMlasPartition {
    /// Complexity-derived target after MLAS caps it to `max_threads * 8`.
    pub target_thread_count: usize,
    /// Number of N/M tiles MLAS passes to `MlasTrySimpleParallel` per GEMM.
    pub threads_per_gemm: usize,
    /// M tile count (`ceil(M / 128)`).
    pub thread_count_m: usize,
    /// N tile count (`ceil(N / stride_n)`).
    pub thread_count_n: usize,
    /// M tile width, fixed by MLAS QNBit.
    pub stride_m: usize,
    /// N tile width after MLAS aligns the complexity-derived split to 16 columns.
    pub stride_n: usize,
    /// Total work-item count passed to the thread-pool callback.
    pub work_items: usize,
    /// ORT `SimpleParallelFor` work items: at most one claimant per pool lane.
    pub ort_claimants: usize,
    /// ORT `LoopCounter` shards: at most eight, capped by pool lanes and blocks.
    pub ort_loop_counter_shards: usize,
}

/// Reproduce the QNBit partition calculation in MLAS's `MlasQNBitGemmBatch`.
pub fn sqnbit_mlas_partitioning(
    m: usize,
    n: usize,
    k: usize,
    batch_n: usize,
    max_threads: usize,
) -> SQNBitMlasPartition {
    const THREAD_COMPLEXITY: usize = 65_536;
    const STRIDE_N_ALIGN: usize = 16;
    const STRIDE_M: usize = 128;
    const MAX_LOOP_COUNTER_SHARDS: usize = 8;

    assert!(batch_n > 0, "batch_n must be non-zero");
    let complexity = m
        .saturating_mul(n)
        .saturating_mul(k)
        .saturating_mul(batch_n);
    let maximum_thread_count = max_threads.max(1).saturating_mul(8);
    let mut target_thread_count = complexity / THREAD_COMPLEXITY + 1;
    target_thread_count = target_thread_count.min(maximum_thread_count);

    let mut threads_per_gemm = (target_thread_count / batch_n).max(1);
    let mut nc = n;
    if threads_per_gemm > 1 {
        let blocked_m = m.div_ceil(STRIDE_M);
        let max_nc = n.saturating_mul(blocked_m).div_ceil(threads_per_gemm);
        if max_nc < nc {
            nc = nc.min(max_nc.div_ceil(STRIDE_N_ALIGN) * STRIDE_N_ALIGN);
        }
    }

    let thread_count_m = m.div_ceil(STRIDE_M);
    let thread_count_n = n.div_ceil(nc);
    threads_per_gemm = thread_count_m * thread_count_n;
    let work_items = threads_per_gemm * batch_n;
    let ort_claimants = max_threads.max(1).min(work_items.max(1));
    let ort_loop_counter_shards = work_items
        .min(MAX_LOOP_COUNTER_SHARDS)
        .min(max_threads.max(1))
        .max(1);

    SQNBitMlasPartition {
        target_thread_count,
        threads_per_gemm,
        thread_count_m,
        thread_count_n,
        stride_m: STRIDE_M,
        stride_n: nc,
        work_items,
        ort_claimants,
        ort_loop_counter_shards,
    }
}

/// Worker count for the standalone MLAS work-stealing pool.
///
/// Precedence, mirroring the CPU EP's own resolution order: the MLAS-specific
/// pool override, then the programmatic budget from
/// [`set_pool_thread_budget`], then the CPU EP's thread-budget environment
/// variable, then [`resolve_default_mlas_threads`].
fn default_mlas_thread_count() -> usize {
    if let Some(threads) = thread_count_env(MLAS_WORK_STEALING_THREADS_ENV) {
        return threads;
    }
    if let Some(threads) = pool_thread_budget() {
        return threads;
    }
    if let Some(threads) = thread_count_env(CPU_DECODE_THREADS_ENV) {
        return threads;
    }

    resolve_default_mlas_threads(
        std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1),
    )
}

/// The worker count the process-global standalone MLAS pool resolves to.
///
/// Callers that interleave their own `rayon` passes with MLAS GEMMs use this to
/// stay inside the same budget: `ONNX_GENAI_CPU_DECODE_THREADS` (and
/// [`set_pool_thread_budget`]) bound MLAS but not `rayon`'s global pool, so a
/// pass that fans out to every logical CPU would silently exceed the width the
/// caller asked for.
///
/// This only *reports* the resolution; it does not create the pool.
pub fn configured_pool_threads() -> usize {
    default_mlas_thread_count().max(1)
}

/// Set or clear a process-local worker budget for the standalone MLAS pool.
///
/// This is the programmatic equivalent of [`CPU_DECODE_THREADS_ENV`] and takes
/// precedence over it. `onnx-runtime-ep-cpu::set_decode_thread_budget` forwards
/// here so that a caller such as the CLI's `--cpu-cores` bounds dense
/// `MlasGemmBatch` work too, on every OS -- not just Linux, where process
/// affinity would otherwise shrink `available_parallelism` for us.
///
/// Pools are initialized lazily and keep their initial size for the process
/// lifetime, so call this before the first GEMM. `Some(0)` is rejected.
pub fn set_pool_thread_budget(threads: Option<usize>) -> Result<(), &'static str> {
    if threads == Some(0) {
        return Err("MLAS pool thread budget must be greater than zero");
    }
    POOL_THREAD_BUDGET.store(threads.unwrap_or(0), Ordering::Release);
    Ok(())
}

fn pool_thread_budget() -> Option<usize> {
    let threads = POOL_THREAD_BUDGET.load(Ordering::Acquire);
    (threads > 0).then(|| clamp_pool_threads(threads))
}

/// The raw programmatic budget, before the [`MAX_MLAS_POOL_THREADS`] cap, or
/// `None` when unset.
///
/// Exposed so higher layers can assert that their own budget actually reached
/// this pool; it reports neither the env-var nor the automatic fallback, and
/// deliberately mirrors the EP's `decode_threads_override` in returning the
/// value as configured rather than as clamped.
pub fn configured_pool_thread_budget() -> Option<usize> {
    std::num::NonZeroUsize::new(POOL_THREAD_BUDGET.load(Ordering::Acquire))
        .map(std::num::NonZeroUsize::get)
}

/// A budget of `0` means "no override" here, matching the CPU EP, which treats
/// `ONNX_GENAI_CPU_DECODE_THREADS=0` as an opt-out back to automatic sizing.
fn thread_count_env(name: &str) -> Option<usize> {
    parse_thread_count(&std::env::var(name).ok()?)
}

/// Pure parser for a thread-count knob, split out so it can be tested without
/// mutating the process environment (`setenv` races other threads' `getenv`).
fn parse_thread_count(raw: &str) -> Option<usize> {
    let threads = raw.trim().parse::<usize>().ok()?;
    (threads > 0).then(|| clamp_pool_threads(threads))
}

fn clamp_pool_threads(threads: usize) -> usize {
    if threads > MAX_MLAS_POOL_THREADS {
        static WARNED: std::sync::Once = std::sync::Once::new();
        WARNED.call_once(|| {
            eprintln!(
                "mlas-sys: thread budget {threads} exceeds the standalone pool cap of \
                 {MAX_MLAS_POOL_THREADS}; using {MAX_MLAS_POOL_THREADS} workers"
            );
        });
    }
    threads.min(MAX_MLAS_POOL_THREADS)
}

/// Default pool size: half the logical CPUs, but never fewer than the previous
/// eight-worker ceiling allowed.
///
/// The old rule was `available.clamp(1, 8)`. That eight-worker cap belongs to
/// the CPU EP's *flat Rayon* pool, whose per-op fork/join regresses past eight
/// workers -- it does not apply here. This pool is persistent and work-stealing,
/// so like the EP's persistent SPMD pool it keeps scaling with cores until the
/// memory-bandwidth knee, measured at about half the logical CPUs.
///
/// Measured on a 32-vCPU/16-core EPYC 9V74 (f32 MatMul K=3584 N=3584 M=128,
/// `native_min` over 15 runs after 5 warmups, ours/ORT at matched threads):
///
/// | requested threads | pool capped at 8 | pool sized to the request |
/// |---|---|---|
/// | 1--8 | 1.00--1.73x | unchanged (cap not binding) |
/// | 16 | 2.08--2.48x slower | 1.24--1.44x slower |
/// | 32 | 1.76--2.38x slower | **0.65--0.82x, i.e. faster than ORT** |
///
/// and with no thread flags at all: 1.85--2.64x slower at the old default
/// versus 0.92--1.30x with this one.
///
/// `max` with `available.min(8)` keeps the rule monotone: no host ever gets
/// *fewer* workers than the old default gave it, so small machines are
/// unaffected and only hosts above 16 logical CPUs change.
fn resolve_default_mlas_threads(available: usize) -> usize {
    let available = available.max(1);
    available
        .div_ceil(2)
        .max(available.min(8))
        .min(MAX_MLAS_POOL_THREADS)
}

fn global_mlas_pool() -> Option<&'static WorkStealingThreadPool> {
    static POOL: OnceLock<Option<WorkStealingThreadPool>> = OnceLock::new();
    POOL.get_or_init(|| WorkStealingThreadPool::new(default_mlas_thread_count()).ok())
        .as_ref()
}

/// Degree of parallelism MLAS sees for internal `TrySimpleParallel` partitioning.
pub fn mlas_threading_degree() -> usize {
    global_mlas_pool().map_or(1, WorkStealingThreadPool::thread_count)
}

/// Work-stealing parallel-for backend for standalone MLAS. MLAS still owns the
/// GEMM tiling/partitioning; this callback only runs MLAS's partition indices on
/// the persistent low-overhead pool instead of Rayon's per-region machinery.
unsafe extern "C" fn work_stealing_parallel_for(
    _rust_ctx: *mut c_void,
    iterations: isize,
    task: MlasTaskFn,
    task_ctx: *mut c_void,
) {
    if iterations <= 0 {
        return;
    }

    MLAS_PARALLEL_FOR_CALLS.fetch_add(1, Ordering::Relaxed);
    MLAS_PARALLEL_FOR_ITERATIONS.fetch_add(iterations as usize, Ordering::Relaxed);

    // Carry the opaque C++ closure pointer across worker threads as an
    // address (usize is Send + Sync). MLAS only *reads* the closure
    // (`std::function::operator() const`) and each `tid` writes a disjoint
    // output partition, so concurrent invocation is race-free.
    let task_ctx = task_ctx as usize;
    if let Some(pool) = global_mlas_pool() {
        let total = iterations as usize;
        MLAS_PARALLEL_FOR_BLOCK_CALLS.fetch_add(1, Ordering::Relaxed);
        MLAS_PARALLEL_FOR_BLOCKS_CLAIMED.fetch_add(total, Ordering::Relaxed);
        pool.parallel_for(0, total, 1, |begin, end| {
            for tid in begin..end {
                // SAFETY: `task_ctx` is valid for the whole MLAS call that
                // drives this parallel-for; each `tid` touches a disjoint
                // partition chosen by MLAS.
                unsafe { task(task_ctx as *mut c_void, tid as isize) };
            }
        });
    } else {
        MLAS_PARALLEL_FOR_FALLBACKS.fetch_add(1, Ordering::Relaxed);
        for tid in 0..iterations {
            unsafe { task(task_ctx as *mut c_void, tid) };
        }
    }
}

/// Report the persistent pool's degree of parallelism to MLAS's partitioner.
unsafe extern "C" fn work_stealing_max_threads(_rust_ctx: *mut c_void) -> c_int {
    mlas_threading_degree().max(1) as c_int
}

static THREADING_INIT: Once = Once::new();

/// Install the work-stealing threading backend into the vendored MLAS build.
/// Idempotent; called before every GEMM entry point. Until this runs (e.g. in
/// the mlas-sys unit tests that call the FFI directly) MLAS stays single
/// threaded, matching the original spike behaviour.
fn ensure_threading() {
    THREADING_INIT.call_once(|| unsafe {
        mlas_set_threading(
            work_stealing_parallel_for,
            work_stealing_max_threads,
            std::ptr::null_mut(),
        );
    });
}

/// Compatibility handle for driving standalone MLAS calls through the
/// `MLAS_THREADPOOL*` parameter.
///
/// In the vendored standalone MLAS build, `MLAS_THREADPOOL` is only a forward
/// declaration of ORT's `onnxruntime::concurrency::ThreadPool` (see
/// `vendor/mlas/onnxruntime/core/mlas/inc/mlas.h`). There is no standalone ORT
/// thread-pool class to construct. Instead, `vendor/shim.cpp` passes a non-null
/// sentinel to APIs such as `MlasQNBitGemmBatch`, and the standalone
/// `MlasGetMaximumThreadCount` / `MlasTrySimpleParallel` hooks route work onto a
/// process-global [`WorkStealingThreadPool`]. This handle no longer creates a
/// per-call pool; it exists for older call sites while the fast path is simply
/// [`sqnbit_gemm`] / [`sqnbit_gemm_with_workspace`] with `multithread=true`.
pub struct MlasThreadPool {
    thread_count: usize,
}

impl MlasThreadPool {
    /// Create a compatibility handle and initialize the process-global backing
    /// pool. The global pool's actual degree of parallelism is selected once at
    /// first use by [`default_mlas_thread_count`] --
    /// `ONNX_GENAI_MLAS_THREADPOOL_THREADS`, then [`set_pool_thread_budget`],
    /// then [`CPU_DECODE_THREADS_ENV`], then
    /// [`resolve_default_mlas_threads`] -- so this `thread_count` is retained
    /// only for diagnostics/backward-compatible tests.
    pub fn new(thread_count: usize) -> std::io::Result<Self> {
        assert!(thread_count > 0, "thread_count must be non-zero");
        ensure_threading();
        let _ = global_mlas_pool();
        Ok(Self {
            thread_count: mlas_threading_degree(),
        })
    }

    /// Number of worker threads reported to MLAS by the global backend.
    pub fn thread_count(&self) -> usize {
        self.thread_count
    }

    fn install<R: Send>(&self, op: impl FnOnce() -> R + Send) -> R {
        op()
    }
}

/// Runtime-selected f32 GEMM microkernel: 512 = AVX-512F, 3 = FMA3/AVX2,
/// 1 = AVX, -1 = other/unknown, 0 = non-x86.
pub fn selected_float_kernel() -> i32 {
    unsafe { mlas_float_kernel_id() as i32 }
}

/// Compute the elementwise logistic (sigmoid) `output = 1 / (1 + exp(-input))`
/// over equal-length contiguous f32 slices using MLAS's SIMD sigmoid. Single
/// threaded; callers shard across threads themselves when needed.
///
/// This is the vectorized primitive behind SiLU (`x * sigmoid(x)`), replacing a
/// scalar `expf` loop that LLVM cannot autovectorize.
pub fn compute_logistic(input: &[f32], output: &mut [f32]) {
    assert_eq!(
        input.len(),
        output.len(),
        "compute_logistic input and output must have equal length"
    );
    if input.is_empty() {
        return;
    }
    // SAFETY: both slices are valid for `n` contiguous f32s; MLAS reads `input`
    // and writes `output`, and Rust's borrow rules prove they do not alias.
    unsafe { mlas_compute_logistic(input.as_ptr(), output.as_mut_ptr(), input.len()) };
}

/// Compute elementwise SiLU `output = input / (1 + exp(-input))` over
/// equal-length contiguous f32 slices. MLAS runtime-dispatches to its fused
/// one-pass AVX-512F kernel when available and uses a portable fallback
/// elsewhere. Single threaded; callers shard across threads themselves.
pub fn compute_silu(input: &[f32], output: &mut [f32]) {
    assert_eq!(
        input.len(),
        output.len(),
        "compute_silu input and output must have equal length"
    );
    if input.is_empty() {
        return;
    }
    // SAFETY: both slices are valid for `n` contiguous f32s; MLAS reads `input`
    // and writes `output`, and Rust's borrow rules prove they do not alias.
    unsafe { mlas_compute_silu(input.as_ptr(), output.as_mut_ptr(), input.len()) };
}

/// Compute elementwise `tanh` over equal-length contiguous f32 slices using
/// MLAS's SIMD polynomial — the same one ONNX Runtime's `Tanh` CPU kernel
/// calls. MLAS dispatches by ISA at runtime, so which kernel runs (and hence
/// the exact bits) depends on the host. Single threaded; callers shard
/// across threads themselves.
pub fn compute_tanh(input: &[f32], output: &mut [f32]) {
    assert_eq!(
        input.len(),
        output.len(),
        "compute_tanh input and output must have equal length"
    );
    if input.is_empty() {
        return;
    }
    // SAFETY: both slices are valid for `n` contiguous f32s; MLAS reads `input`
    // and writes `output`, and Rust's borrow rules prove they do not alias.
    unsafe { mlas_compute_tanh(input.as_ptr(), output.as_mut_ptr(), input.len()) };
}

/// Compute elementwise `erf` over equal-length contiguous f32 slices using
/// MLAS's SIMD polynomial — the same one ONNX Runtime's `Erf` CPU kernel
/// calls. Which kernel runs depends on MLAS's runtime ISA dispatch.
/// Single threaded; callers shard across threads themselves.
pub fn compute_erf(input: &[f32], output: &mut [f32]) {
    assert_eq!(
        input.len(),
        output.len(),
        "compute_erf input and output must have equal length"
    );
    if input.is_empty() {
        return;
    }
    // SAFETY: both slices are valid for `n` contiguous f32s; MLAS reads `input`
    // and writes `output`, and Rust's borrow rules prove they do not alias.
    unsafe { mlas_compute_erf(input.as_ptr(), output.as_mut_ptr(), input.len()) };
}

/// Compute elementwise exact GELU `x * 0.5 * (1 + erf(x / sqrt(2)))` over
/// equal-length contiguous f32 slices, fused in one MLAS pass.
///
/// MLAS requires that input and output not overlap (`mlas.h:1166`); the
/// `&[f32]` / `&mut [f32]` signature makes that unrepresentable.
///
/// Single threaded; callers shard across threads themselves.
pub fn compute_gelu_erf(input: &[f32], output: &mut [f32]) {
    assert_eq!(
        input.len(),
        output.len(),
        "compute_gelu_erf input and output must have equal length"
    );
    if input.is_empty() {
        return;
    }
    // SAFETY: both slices are valid for `n` contiguous f32s; MLAS reads `input`
    // and writes `output`, and Rust's borrow rules prove they do not alias —
    // which is also what discharges MLAS's no-overlap precondition.
    unsafe { mlas_compute_gelu_erf(input.as_ptr(), output.as_mut_ptr(), input.len()) };
}

/// Row-wise in-place softmax, replacing a scalar `expf` loop: normalizes `n` rows of
/// `d` contiguous f32s with MLAS's SIMD max reduction and polynomial exp — the
/// same primitive ONNX Runtime's own Softmax and attention kernels use.
///
/// Single threaded; callers shard across threads themselves.
///
/// A row consisting entirely of `-inf` produces NaN, matching MLAS/ORT; callers
/// that need the "fully masked row → zero" convention must screen for that case
/// themselves.
pub fn compute_softmax_in_place(data: &mut [f32], n: usize, d: usize) {
    assert_eq!(
        data.len(),
        n.saturating_mul(d),
        "compute_softmax_in_place expects exactly n*d elements"
    );
    if data.is_empty() {
        return;
    }
    // SAFETY: `data` holds `n * d` contiguous f32s (asserted above) and is
    // uniquely borrowed. MLAS's non-log softmax streams each row forward from
    // input to output and then rescales the output, so a single buffer serving
    // as both is well defined.
    unsafe { mlas_compute_softmax_in_place(data.as_mut_ptr(), n, d) };
}

/// Compute contiguous Float32 elementwise addition with MLAS SIMD.
pub fn eltwise_add(left: &[f32], right: &[f32], output: &mut [f32]) {
    assert_eq!(left.len(), right.len());
    assert_eq!(left.len(), output.len());
    unsafe {
        mlas_eltwise_add(
            left.as_ptr(),
            right.as_ptr(),
            output.as_mut_ptr(),
            output.len(),
        );
    }
}

/// Compute contiguous Float32 ReLU with MLAS SIMD.
pub fn compute_relu(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());
    unsafe {
        mlas_compute_activation(
            1,
            0.0,
            0.0,
            input.as_ptr(),
            output.as_mut_ptr(),
            output.len(),
        );
    }
}

/// Compute contiguous Float32 clipping with MLAS SIMD.
pub fn compute_clip(input: &[f32], output: &mut [f32], minimum: f32, maximum: f32) {
    assert_eq!(input.len(), output.len());
    unsafe {
        mlas_compute_activation(
            5,
            minimum,
            maximum,
            input.as_ptr(),
            output.as_mut_ptr(),
            output.len(),
        );
    }
}

/// Prepared MLAS Float32 convolution parameters for one concrete NCHW shape.
pub struct ConvPlan {
    ptr: NonNull<c_void>,
    working_buffer_elements: usize,
}

// SAFETY: MLAS treats prepared convolution parameters as immutable during
// execution. Each call supplies disjoint input, scratch, and output buffers.
unsafe impl Send for ConvPlan {}
unsafe impl Sync for ConvPlan {}

impl ConvPlan {
    /// Prepare an N-dimensional NCHW convolution and return its scratch size.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        batch_count: usize,
        group_count: usize,
        input_channels_per_group: usize,
        input_shape: &[i64],
        kernel_shape: &[i64],
        dilation_shape: &[i64],
        padding: &[i64],
        stride_shape: &[i64],
        output_shape: &[i64],
        filter_count_per_group: usize,
    ) -> Option<Self> {
        let dimensions = input_shape.len();
        assert!((1..=3).contains(&dimensions));
        assert_eq!(kernel_shape.len(), dimensions);
        assert_eq!(dilation_shape.len(), dimensions);
        assert_eq!(padding.len(), dimensions * 2);
        assert_eq!(stride_shape.len(), dimensions);
        assert_eq!(output_shape.len(), dimensions);
        ensure_threading();
        let mut working_buffer_elements = 0;
        let ptr = unsafe {
            mlas_conv_prepare(
                dimensions,
                batch_count,
                group_count,
                input_channels_per_group,
                input_shape.as_ptr(),
                kernel_shape.as_ptr(),
                dilation_shape.as_ptr(),
                padding.as_ptr(),
                stride_shape.as_ptr(),
                output_shape.as_ptr(),
                filter_count_per_group,
                &mut working_buffer_elements,
            )
        };
        Some(Self {
            ptr: NonNull::new(ptr)?,
            working_buffer_elements,
        })
    }

    /// Number of Float32 scratch elements required by [`Self::run`].
    pub fn working_buffer_elements(&self) -> usize {
        self.working_buffer_elements
    }

    /// Execute the prepared convolution.
    pub fn run(
        &self,
        input: &[f32],
        filter: &[f32],
        bias: Option<&[f32]>,
        working_buffer: &mut [f32],
        output: &mut [f32],
    ) {
        assert!(working_buffer.len() >= self.working_buffer_elements);
        ensure_threading();
        unsafe {
            mlas_conv_run(
                self.ptr.as_ptr(),
                input.as_ptr(),
                filter.as_ptr(),
                bias.map_or(std::ptr::null(), <[f32]>::as_ptr),
                if self.working_buffer_elements == 0 {
                    std::ptr::null_mut()
                } else {
                    working_buffer.as_mut_ptr()
                },
                output.as_mut_ptr(),
            );
        }
    }
}

impl Drop for ConvPlan {
    fn drop(&mut self) {
        unsafe { mlas_conv_plan_destroy(self.ptr.as_ptr()) };
    }
}

/// MLAS activation applied inside (or immediately after) a convolution.
///
/// Mirrors `MLAS_ACTIVATION_KIND`; the two `values` carry the kind-specific
/// parameters (`Clip` min/max, `LeakyRelu`/`HardSigmoid` alpha/beta).
#[derive(Clone, Copy, Debug)]
pub struct NchwcActivation {
    /// Raw `MLAS_ACTIVATION_KIND` discriminant (0 = identity, 1 = relu, 5 = clip).
    pub kind: i32,
    /// Kind-specific parameters, laid over the `Parameters.Values[2]` union.
    pub values: [f32; 2],
}

impl NchwcActivation {
    /// No activation (`MlasIdentityActivation`).
    pub const IDENTITY: Self = Self {
        kind: 0,
        values: [0.0, 0.0],
    };
    /// ReLU (`MlasReluActivation`).
    pub const RELU: Self = Self {
        kind: 1,
        values: [0.0, 0.0],
    };

    /// Clip activation (`MlasClipActivation`) with the given bounds.
    pub fn clip(minimum: f32, maximum: f32) -> Self {
        Self {
            kind: 5,
            values: [minimum, maximum],
        }
    }
}

/// SIMD channel-block width used by the MLAS NCHWc kernels (8 for AVX2, 16 for
/// AVX-512). A value `<= 1` means the host has no blocked-convolution kernel and
/// callers must use the plain [`ConvPlan`] path instead.
pub fn nchwc_block_size() -> usize {
    unsafe { mlas_nchwc_block_size() }
}

/// Reorder an `OIHW` filter into the `OIHWBiBo` layout (both input and output
/// channels blocked), padding partial blocks with zeros. `dest` must hold
/// `round_up(O, block) * round_up(I, block) * H * W` elements.
pub fn nchwc_reorder_filter_bibo(filter_shape: &[i64; 4], source: &[f32], dest: &mut [f32]) {
    unsafe {
        mlas_nchwc_reorder_filter_bibo(filter_shape.as_ptr(), source.as_ptr(), dest.as_mut_ptr())
    };
}

/// Reorder an `OIHW` filter into the `OIHWBo` layout (only output channels
/// blocked), padding partial output blocks with zeros. Used for the NCHW-input
/// (first-layer) and depthwise algorithms. `dest` must hold
/// `round_up(O, block) * I * H * W` elements.
pub fn nchwc_reorder_filter_bo(filter_shape: &[i64; 4], source: &[f32], dest: &mut [f32]) {
    unsafe {
        mlas_nchwc_reorder_filter_bo(filter_shape.as_ptr(), source.as_ptr(), dest.as_mut_ptr())
    };
}

/// Reorder an NCHW activation plane set into NCHWc. `channels` must be a
/// multiple of 4; `dest` must hold `round_up(channels, block) * input_size`
/// elements (partial trailing block is zero padded).
pub fn nchwc_reorder_input_nchw(
    source: &[f32],
    dest: &mut [f32],
    channels: usize,
    input_size: usize,
) {
    ensure_threading();
    unsafe {
        mlas_nchwc_reorder_input_nchw(source.as_ptr(), dest.as_mut_ptr(), channels, input_size)
    };
}

/// Reorder an NCHWc output buffer back to dense NCHW, keeping only
/// `output_shape[1]` channels.
pub fn nchwc_reorder_output_nchw(output_shape: &[i64; 4], source: &[f32], dest: &mut [f32]) {
    ensure_threading();
    unsafe {
        mlas_nchwc_reorder_output_nchw(output_shape.as_ptr(), source.as_ptr(), dest.as_mut_ptr())
    };
}

/// Execute an NCHWc blocked 2-D convolution.
///
/// `input`/`output` are in NCHWc block layout except for the NCHW-input
/// (first-layer) algorithm, where `input` stays plain NCHW. `filter` must be
/// pre-reordered (`OIHWBiBo` or `OIHWBo`) to match the algorithm MLAS selects
/// from the shape. `bias`, when present, must be padded to
/// `round_up(output_channels, block)` elements. `zero_mode` false accumulates
/// into `output` (Conv/Sum fusion); true overwrites it.
#[allow(clippy::too_many_arguments)]
pub fn nchwc_conv(
    input_shape: &[i64; 4],
    kernel_shape: &[i64; 2],
    dilation_shape: &[i64; 2],
    padding: &[i64; 4],
    stride_shape: &[i64; 2],
    output_shape: &[i64; 4],
    group_count: usize,
    input: &[f32],
    filter: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    activation: NchwcActivation,
    zero_mode: bool,
) {
    ensure_threading();
    unsafe {
        mlas_nchwc_conv(
            input_shape.as_ptr(),
            kernel_shape.as_ptr(),
            dilation_shape.as_ptr(),
            padding.as_ptr(),
            stride_shape.as_ptr(),
            output_shape.as_ptr(),
            group_count,
            input.as_ptr(),
            filter.as_ptr(),
            bias.map_or(std::ptr::null(), <[f32]>::as_ptr),
            output.as_mut_ptr(),
            activation.kind,
            activation.values[0],
            activation.values[1],
            i32::from(zero_mode),
        );
    }
}

/// MLAS Float32 pooling mode.
#[derive(Clone, Copy, Debug)]
#[repr(i32)]
pub enum PoolKind {
    Maximum = 0,
    AverageExcludePad = 1,
    AverageIncludePad = 2,
}

/// Execute an N-dimensional NCHW Float32 pool using MLAS.
#[allow(clippy::too_many_arguments)]
pub fn pool(
    kind: PoolKind,
    input_shape: &[i64],
    kernel_shape: &[i64],
    padding: &[i64],
    stride_shape: &[i64],
    output_shape: &[i64],
    input: &[f32],
    output: &mut [f32],
) {
    let dimensions = input_shape.len().saturating_sub(2);
    assert!((1..=3).contains(&dimensions));
    assert_eq!(kernel_shape.len(), dimensions);
    assert_eq!(padding.len(), dimensions * 2);
    assert_eq!(stride_shape.len(), dimensions);
    assert_eq!(output_shape.len(), dimensions + 2);
    ensure_threading();
    unsafe {
        mlas_pool(
            kind as c_int,
            dimensions,
            input_shape.as_ptr(),
            kernel_shape.as_ptr(),
            padding.as_ptr(),
            stride_shape.as_ptr(),
            output_shape.as_ptr(),
            input.as_ptr(),
            output.as_mut_ptr(),
        );
    }
}

/// Execute an NCHWc blocked 2-D pool using MLAS.
///
/// `input_shape` / `output_shape` are the blocked NCHWc shapes
/// `[N, round_up(C, block), H, W]`; MLAS pools each channel independently on the
/// blocked buffer, so callers keep the activation in NCHWc across the pool with
/// no reorder. `input` / `output` are blocked buffers. Mirrors ONNX Runtime's
/// `NchwcTransformer` handling of pooling.
#[allow(clippy::too_many_arguments)]
pub fn nchwc_pool(
    kind: PoolKind,
    input_shape: &[i64; 4],
    kernel_shape: &[i64; 2],
    dilation_shape: &[i64; 2],
    padding: &[i64; 4],
    stride_shape: &[i64; 2],
    output_shape: &[i64; 4],
    input: &[f32],
    output: &mut [f32],
) {
    ensure_threading();
    unsafe {
        mlas_nchwc_pool(
            kind as c_int,
            input_shape.as_ptr(),
            kernel_shape.as_ptr(),
            dilation_shape.as_ptr(),
            padding.as_ptr(),
            stride_shape.as_ptr(),
            output_shape.as_ptr(),
            input.as_ptr(),
            output.as_mut_ptr(),
        );
    }
}
///
/// MLAS's packed layout is accessed with aligned AVX-512 loads/stores, so the
/// backing allocation is 64-byte aligned (a plain `Vec<u8>` is not).
pub struct PackedB {
    ptr: *mut u8,
    layout: std::alloc::Layout,
    n: usize,
    k: usize,
}

// SAFETY: construction fully initializes the allocation, which is immutable
// afterward. Packed GEMM calls only read it, so shared concurrent use is safe.
unsafe impl Send for PackedB {}
unsafe impl Sync for PackedB {}

impl PackedB {
    /// Pack a row-major `k x n` B matrix (no transpose, `ldb = n`).
    pub fn new(n: usize, k: usize, b: &[f32]) -> Self {
        assert_eq!(b.len(), k * n);
        let size = unsafe { mlas_sgemm_pack_b_size(0, 0, n, k) }.max(1);
        let layout = std::alloc::Layout::from_size_align(size, 64).unwrap();
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!ptr.is_null(), "packed-B allocation failed");
        unsafe { mlas_sgemm_pack_b(0, 0, n, k, b.as_ptr(), n, ptr) };
        Self { ptr, layout, n, k }
    }

    /// Return the logical `(k, n)` dimensions of the packed B matrix.
    pub fn dimensions(&self) -> (usize, usize) {
        (self.k, self.n)
    }
}

impl Drop for PackedB {
    fn drop(&mut self) {
        unsafe { std::alloc::dealloc(self.ptr, self.layout) };
    }
}

/// `C = A * packed(B)` for row-major A (`m x k`), reusing a pre-packed B.
pub fn sgemm_nn_packed(m: usize, a: &[f32], packed: &PackedB, c: &mut [f32]) {
    let (n, k) = (packed.n, packed.k);
    assert_eq!(a.len(), m * k);
    assert_eq!(c.len(), m * n);
    ensure_threading();
    unsafe {
        mlas_sgemm_packed(
            0,
            0,
            m,
            n,
            k,
            1.0,
            a.as_ptr(),
            k,
            packed.ptr,
            0.0,
            c.as_mut_ptr(),
            n,
        );
    }
}

/// Zero points for an integer GEMM operand.
#[derive(Debug, Clone, Copy)]
pub enum QgemmZeroPoints<'a> {
    /// One zero point for the whole tensor.
    PerTensor(u8),
    /// One zero point per column of `B`, i.e. `n` entries.
    PerColumn(&'a [u8]),
}

/// Row-major integer GEMM producing the `i32` accumulator, `C = (A - za) * (B - zb)`.
///
/// `a` is `m x k` with row stride `k`, `b` is `k x n` with row stride `n`, and
/// `c` is `m x n` with row stride `n` -- the same layout ONNX Runtime hands to
/// MLAS for `MatMulInteger`/`QLinearMatMul`. Requantization is left to the
/// caller, so the result is exactly the integer dot product with the zero
/// points folded in.
///
/// `a_signed` / `b_signed` select the `i8` interpretation of the respective
/// operand; the bytes are passed through unchanged either way. Zero-point bytes
/// follow the same interpretation as the operand they belong to.
///
/// # Panics
///
/// Panics if any slice length disagrees with `m`, `n`, `k`, or if a per-column
/// zero-point slice is not exactly `n` long.
#[allow(clippy::too_many_arguments)]
pub fn qgemm_i32(
    m: usize,
    n: usize,
    k: usize,
    a: &[u8],
    a_signed: bool,
    zero_point_a: u8,
    b: &[u8],
    b_signed: bool,
    zero_point_b: QgemmZeroPoints<'_>,
    c: &mut [i32],
) {
    assert_eq!(a.len(), m * k, "A must be m*k bytes");
    assert_eq!(b.len(), k * n, "B must be k*n bytes");
    assert_eq!(c.len(), m * n, "C must be m*n i32");
    if m == 0 || n == 0 {
        return;
    }
    let (zero_point_b_ptr, per_column) = match zero_point_b {
        QgemmZeroPoints::PerTensor(ref value) => (std::ptr::from_ref(value), 0),
        QgemmZeroPoints::PerColumn(values) => {
            assert_eq!(values.len(), n, "per-column zero points must be n long");
            (values.as_ptr(), 1)
        }
    };
    ensure_threading();
    // SAFETY: the three slices are exactly the sizes asserted above, the zero
    // point pointer is either a per-column slice of length `n` or a borrow of a
    // local `u8` that outlives the call, and the shim writes only through `c`.
    unsafe {
        mlas_qgemm_i32(
            m,
            n,
            k,
            c_int::from(a_signed),
            c_int::from(b_signed),
            a.as_ptr().cast::<c_void>(),
            k,
            zero_point_a,
            b.as_ptr().cast::<c_void>(),
            n,
            zero_point_b_ptr,
            per_column,
            c.as_mut_ptr(),
            n,
            c_int::from(mlas_threading_degree() > 1),
        );
    }
}

/// A constant quantized `B` pre-packed into MLAS's kernel layout.
///
/// [`qgemm_i32`] leaves `BIsPacked` unset, so MLAS re-packs the whole `k x n`
/// weight on every call. ORT pre-packs a constant weight once at session
/// initialisation instead; this is the same thing.
///
/// The packed layout is chosen by the kernel MLAS dispatches to, so a pack is
/// only valid for the same `(n, k, a_signed, b_signed)` **on the same machine**.
/// It must never be serialised, cached to disk, or shared across processes.
pub struct QgemmPackedB {
    ptr: *mut u8,
    layout: std::alloc::Layout,
    n: usize,
    k: usize,
    a_signed: bool,
    b_signed: bool,
}

// SAFETY: the buffer is written once during `new` and only ever read afterwards
// (MLAS takes it as `const void*`), so sharing it across threads is sound. This
// mirrors `PackedB` above.
unsafe impl Send for QgemmPackedB {}
unsafe impl Sync for QgemmPackedB {}

/// Live heap bytes currently retained by all [`QgemmPackedB`] instances -- the
/// MLAS packed allocation only (there is no owned scale/zp copy for the integer
/// GEMM). Maintained by construction/`Drop` so a caller can compare the memory
/// plan's *predicted* packed footprint against the bytes the kernels *actually*
/// hold, in the same run. Mirrors [`SQNBIT_PACKED_LIVE_BYTES`].
static QGEMM_PACKED_LIVE_BYTES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Snapshot of [`QGEMM_PACKED_LIVE_BYTES`]: the heap bytes all live
/// [`QgemmPackedB`] instances currently retain.
pub fn qgemm_packed_live_bytes() -> usize {
    QGEMM_PACKED_LIVE_BYTES.load(std::sync::atomic::Ordering::Relaxed)
}

/// The exact MLAS packed-B buffer size for a constant quantized B, or `None`
/// when MLAS reports no packed layout for this shape/signedness on the current
/// host (the caller then uses the unpacked path). Lets the memory plan predict
/// the pre-pack footprint before any weight is packed.
pub fn qgemm_pack_b_size(n: usize, k: usize, a_signed: bool, b_signed: bool) -> Option<usize> {
    if n == 0 || k == 0 {
        return None;
    }
    // SAFETY: a pure size query with no pointer arguments.
    let size =
        unsafe { mlas_qgemm_pack_b_size(n, k, c_int::from(a_signed), c_int::from(b_signed)) };
    (size != 0).then_some(size)
}

impl QgemmPackedB {
    /// Pack a row-major `k x n` quantized B (`ldb == n`).
    ///
    /// Returns `None` when MLAS reports no packed layout for this shape and
    /// signedness combination, which is its documented way of saying "call the
    /// unpacked path"; callers must then keep using [`qgemm_i32`].
    pub fn new(n: usize, k: usize, b: &[u8], a_signed: bool, b_signed: bool) -> Option<Self> {
        assert_eq!(b.len(), k * n, "B must be k*n bytes");
        if n == 0 || k == 0 {
            return None;
        }
        // SAFETY: a pure size query with no pointer arguments.
        let size =
            unsafe { mlas_qgemm_pack_b_size(n, k, c_int::from(a_signed), c_int::from(b_signed)) };
        if size == 0 {
            return None;
        }
        let layout = std::alloc::Layout::from_size_align(size, 64).ok()?;
        // SAFETY: `size` is non-zero, so the layout is non-zero-sized.
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!ptr.is_null(), "packed quantized-B allocation failed");
        // SAFETY: `b` is `k * n` bytes as asserted, `ldb = n` matches the
        // row-major layout, and `ptr` addresses `size` writable bytes -- the
        // exact size MLAS just asked for.
        unsafe {
            mlas_qgemm_pack_b(
                n,
                k,
                b.as_ptr().cast::<c_void>(),
                n,
                c_int::from(a_signed),
                c_int::from(b_signed),
                ptr.cast::<c_void>(),
            );
        }
        QGEMM_PACKED_LIVE_BYTES.fetch_add(layout.size(), std::sync::atomic::Ordering::Relaxed);
        Some(Self {
            ptr,
            layout,
            n,
            k,
            a_signed,
            b_signed,
        })
    }

    /// Heap bytes this pack retains: the MLAS packed allocation.
    pub fn owned_heap_bytes(&self) -> usize {
        self.layout.size()
    }

    /// The logical `(k, n)` dimensions and `(a_signed, b_signed)` this pack was
    /// built for. A caller must not use it for any other combination.
    pub fn identity(&self) -> (usize, usize, bool, bool) {
        (self.k, self.n, self.a_signed, self.b_signed)
    }
}

impl Drop for QgemmPackedB {
    fn drop(&mut self) {
        QGEMM_PACKED_LIVE_BYTES.fetch_sub(
            self.owned_heap_bytes(),
            std::sync::atomic::Ordering::Relaxed,
        );
        // SAFETY: `ptr`/`layout` are the pair returned by `alloc_zeroed` in
        // `new` and this runs at most once.
        unsafe { std::alloc::dealloc(self.ptr, self.layout) };
    }
}

/// [`qgemm_i32`] against a B that was pre-packed by [`QgemmPackedB::new`].
///
/// Panics if `packed` was built for a different shape or signedness, which
/// would otherwise make MLAS read the wrong layout.
#[allow(clippy::too_many_arguments)]
pub fn qgemm_i32_packed(
    m: usize,
    n: usize,
    k: usize,
    a: &[u8],
    a_signed: bool,
    zero_point_a: u8,
    packed: &QgemmPackedB,
    zero_point_b: QgemmZeroPoints<'_>,
    c: &mut [i32],
) {
    assert_eq!(a.len(), m * k, "A must be m*k bytes");
    assert_eq!(c.len(), m * n, "C must be m*n i32");
    assert_eq!(
        packed.identity(),
        (k, n, a_signed, packed.b_signed),
        "the packed B was built for a different shape or signedness"
    );
    if m == 0 || n == 0 {
        return;
    }
    let (zero_point_b_ptr, per_column) = match zero_point_b {
        QgemmZeroPoints::PerTensor(ref value) => (std::ptr::from_ref(value), 0),
        QgemmZeroPoints::PerColumn(values) => {
            assert_eq!(values.len(), n, "per-column zero points must be n long");
            (values.as_ptr(), 1)
        }
    };
    ensure_threading();
    // SAFETY: `a` and `c` are exactly the sizes asserted above, `packed` owns a
    // buffer MLAS itself sized and filled for this `(n, k, signedness)`, the
    // zero point pointer is either an `n`-long slice or a borrow of a local that
    // outlives the call, and the shim writes only through `c`.
    unsafe {
        mlas_qgemm_i32_packed(
            m,
            n,
            k,
            c_int::from(a_signed),
            c_int::from(packed.b_signed),
            a.as_ptr().cast::<c_void>(),
            k,
            zero_point_a,
            packed.ptr.cast::<c_void>(),
            zero_point_b_ptr,
            per_column,
            c.as_mut_ptr(),
            n,
            c_int::from(mlas_threading_degree() > 1),
        );
    }
}

/// The `B` operand of a requantizing quantized GEMM: either raw `k x n` bytes
/// or a pack built once by [`QgemmPackedB::new`].
#[derive(Clone, Copy)]
pub enum QgemmWeights<'a> {
    /// Row-major `k x n` bytes with row stride `n`.
    Dense {
        /// The weight bytes.
        bytes: &'a [u8],
        /// Whether the bytes are `i8` rather than `u8`.
        signed: bool,
    },
    /// A pack MLAS built for this `(n, k, a_signed, b_signed)` on this machine.
    Packed(&'a QgemmPackedB),
}

/// Output scale of a requantizing quantized GEMM.
#[derive(Clone, Copy)]
pub enum QgemmScale<'a> {
    /// One scale for the whole result.
    PerTensor(f32),
    /// One scale per column of the result, i.e. `n` entries.
    PerColumn(&'a [f32]),
}

/// Integer GEMM whose accumulator is requantized to bytes **inside** MLAS.
///
/// [`qgemm_i32`] returns the raw `i32` accumulator, which leaves the caller to
/// walk the whole `m x n` array a second time to scale, round, offset and
/// narrow it. ONNX Runtime does not do that: its `QLinearMatMul` passes MLAS a
/// `MLAS_QGEMM_REQUANT_OUTPUT_PROCESSOR`, so each output tile is requantized as
/// soon as the kernel produces it, while it is still in cache, and the final
/// bytes land straight in the destination tensor. This is that path.
///
/// `c` is scratch: MLAS accumulates into it and requantizes it in place, so it
/// must be `m * n` long, but its contents afterwards are unspecified and it
/// does **not** need to be zeroed first.
///
/// Numerically the processor computes, per element,
/// `clamp(round_ties_even(c * scale), lo - zp, hi - zp) + zp`, where `lo`/`hi`
/// are the output dtype's bounds. Clamping the *float* before rounding and
/// rounding before clamping agree for every finite input, because the clamp
/// bounds are integers -- but they disagree on `NaN`, which this path maps to
/// `lo` and a `round`-then-clamp scalar loop maps to `zp`. A caller that needs
/// bit-identity with such a loop must therefore keep non-finite scales off this
/// path; a finite scale can only produce `NaN` from a non-finite accumulator,
/// which `i32` cannot hold.
///
/// # Panics
///
/// Panics if any slice length disagrees with `m`, `n`, `k`, if a per-column
/// scale or zero point is not exactly `n` long, or if a pack was built for a
/// different shape or signedness.
#[allow(clippy::too_many_arguments)]
pub fn qgemm_requantize(
    m: usize,
    n: usize,
    k: usize,
    a: &[u8],
    a_signed: bool,
    zero_point_a: u8,
    b: QgemmWeights<'_>,
    zero_point_b: QgemmZeroPoints<'_>,
    scale: QgemmScale<'_>,
    output: &mut [u8],
    output_signed: bool,
    output_zero_point: i32,
    c: &mut [i32],
) {
    assert_eq!(a.len(), m * k, "A must be m*k bytes");
    assert_eq!(c.len(), m * n, "C must be m*n i32");
    assert_eq!(output.len(), m * n, "output must be m*n bytes");
    if m == 0 || n == 0 {
        return;
    }
    let (b_ptr, b_signed, ldb, b_is_packed) = match b {
        QgemmWeights::Dense { bytes, signed } => {
            assert_eq!(bytes.len(), k * n, "B must be k*n bytes");
            (bytes.as_ptr().cast::<c_void>(), signed, n, 0)
        }
        QgemmWeights::Packed(packed) => {
            assert_eq!(
                packed.identity(),
                (k, n, a_signed, packed.b_signed),
                "the packed B was built for a different shape or signedness"
            );
            (
                packed.ptr.cast_const().cast::<c_void>(),
                packed.b_signed,
                n,
                1,
            )
        }
    };
    let (zero_point_b_ptr, per_column_zero_points) = match zero_point_b {
        QgemmZeroPoints::PerTensor(ref value) => (std::ptr::from_ref(value), 0),
        QgemmZeroPoints::PerColumn(values) => {
            assert_eq!(values.len(), n, "per-column zero points must be n long");
            (values.as_ptr(), 1)
        }
    };
    let (scale_ptr, per_column_scale) = match scale {
        QgemmScale::PerTensor(ref value) => (std::ptr::from_ref(value), 0),
        QgemmScale::PerColumn(values) => {
            assert_eq!(values.len(), n, "per-column scales must be n long");
            (values.as_ptr(), 1)
        }
    };
    ensure_threading();
    // SAFETY: every slice is exactly the size asserted above; the zero-point and
    // scale pointers are either `n`-long slices or borrows of locals that
    // outlive the call; a packed `B` owns a buffer MLAS itself sized and filled
    // for this `(n, k, signedness)`; and the shim writes only through `c` and
    // `output`, which are distinct `&mut` borrows.
    unsafe {
        mlas_qgemm_requantize(
            m,
            n,
            k,
            c_int::from(a_signed),
            c_int::from(b_signed),
            a.as_ptr().cast::<c_void>(),
            k,
            zero_point_a,
            b_ptr,
            ldb,
            b_is_packed,
            zero_point_b_ptr,
            per_column_zero_points,
            c.as_mut_ptr(),
            n,
            output.as_mut_ptr().cast::<c_void>(),
            n,
            c_int::from(output_signed),
            scale_ptr,
            per_column_scale,
            output_zero_point,
            c_int::from(mlas_threading_degree() > 1),
        );
    }
}

/// Whether this machine's `u8 x i8` integer GEMM kernel is exact.
///
/// On AVX2 without VNNI, MLAS's `u8 x i8` kernel multiplies raw bytes with
/// `vpmaddubsw`, which sums *adjacent pairs* into an `i16` **with saturation**:
/// `255 * -128 + 255 * -128 == -65280` clamps to `-32768`. Zero points are
/// applied afterwards as row/column corrections, so the saturation is not
/// recoverable and the result is silently approximate. VNNI (`vpdpbusd`) and
/// AMX accumulate straight into `i32` and do not saturate; the other three
/// signedness pairs are exact on every kernel we ship.
///
/// Rather than hard-code an ISA table -- which would be wrong on the next
/// microarchitecture in either direction -- probe once with an input
/// constructed to saturate, and believe the answer. A caller that requires bit
/// exactness must consult this before routing `u8 x i8` here.
pub fn qgemm_u8s8_is_exact() -> bool {
    static EXACT: OnceLock<bool> = OnceLock::new();
    *EXACT.get_or_init(|| {
        // (255 * -128) + (255 * -128) = -65280, which is outside i16 range, so a
        // pairwise-i16 kernel returns the clamped -32768 instead.
        let a = [255u8, 255];
        let b = [0x80u8, 0x80];
        let mut c = [0i32];
        qgemm_i32(
            1,
            1,
            2,
            &a,
            false,
            0,
            &b,
            true,
            QgemmZeroPoints::PerTensor(0),
            &mut c,
        );
        c[0] == -65280
    })
}

/// Safe wrapper computing `C = A * B` for row-major matrices with no transpose.
///
/// `a` is `m x k`, `b` is `k x n`, `c` is `m x n`. Uses `alpha = 1`,
/// `beta = 0` (C is overwritten).
pub fn sgemm_nn(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    assert_eq!(a.len(), m * k, "A must be m*k");
    assert_eq!(b.len(), k * n, "B must be k*n");
    assert_eq!(c.len(), m * n, "C must be m*n");
    ensure_threading();
    unsafe {
        mlas_sgemm(
            0,
            0,
            m,
            n,
            k,
            1.0,
            a.as_ptr(),
            k,
            b.as_ptr(),
            n,
            0.0,
            c.as_mut_ptr(),
            n,
        );
    }
}

/// General entry point mirroring the C shim, exposing transpose flags and
/// alpha/beta. Leading dimensions default to the natural row-major strides.
#[allow(clippy::too_many_arguments)]
pub fn sgemm(
    trans_a: bool,
    trans_b: bool,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    beta: f32,
    c: &mut [f32],
    ldc: usize,
) {
    ensure_threading();
    unsafe {
        mlas_sgemm(
            trans_a as c_int,
            trans_b as c_int,
            m,
            n,
            k,
            alpha,
            a.as_ptr(),
            lda,
            b.as_ptr(),
            ldb,
            beta,
            c.as_mut_ptr(),
            ldc,
        );
    }
}

/// One GEMM inside a batched SGEMM. `M`, `N`, `K`, the transpose flags and
/// every leading dimension are shared across the batch; only the operands and
/// the output window differ.
#[derive(Clone, Copy, Debug)]
pub struct SgemmBatchItem<'a> {
    /// `A` operand, at least `m * lda` floats (`k * lda` when `trans_a`).
    pub a: &'a [f32],
    /// `B` operand, at least `k * ldb` floats (`n * ldb` when `trans_b`).
    pub b: &'a [f32],
    /// Element offset of this item's `m * ldc` output window inside `c`.
    pub c_offset: usize,
}

/// Batched row-major SGEMM: `C_i = alpha * op(A_i) * op(B_i) + beta * C_i`.
///
/// The point is not to save per-call overhead in C++, which is trivial, but to
/// hand MLAS a single `MlasTrySimpleParallel` fan-out covering the whole batch.
/// Issued one at a time, each small GEMM takes the work-stealing pool's
/// dispatch lock and asks for a thread count derived from its own complexity
/// alone; batched, MLAS spreads `ThreadsPerGemm * batch` work items across the
/// pool in one dispatch (`sgemm.cpp`'s `MlasGemmBatch`). A grouped
/// mixture-of-experts decode step is exactly this shape: `k` skinny GEMMs that
/// individually look far too small to thread.
///
/// # Panics
///
/// Panics if any operand is too short for the declared dimensions, or if the
/// items' output windows are not strictly ascending and disjoint - the
/// invariant that makes handing out several `*mut f32` into one `&mut [f32]`
/// sound.
#[allow(clippy::too_many_arguments)]
pub fn sgemm_batch(
    trans_a: bool,
    trans_b: bool,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    items: &[SgemmBatchItem<'_>],
    lda: usize,
    ldb: usize,
    beta: f32,
    c: &mut [f32],
    ldc: usize,
) {
    if items.is_empty() || m == 0 || n == 0 || k == 0 {
        return;
    }
    assert!(
        lda >= if trans_a { m } else { k },
        "sgemm_batch: lda {lda} is smaller than the packed row length"
    );
    assert!(
        ldb >= if trans_b { k } else { n },
        "sgemm_batch: ldb {ldb} is smaller than the packed row length"
    );
    assert!(ldc >= n, "sgemm_batch: ldc {ldc} is smaller than n {n}");
    let a_min = if trans_a { k * lda } else { m * lda };
    let b_min = if trans_b { n * ldb } else { k * ldb };
    let c_window = m * ldc;
    let mut previous_end = 0usize;
    for (index, item) in items.iter().enumerate() {
        assert!(
            item.a.len() >= a_min,
            "batched SGEMM item {index} A holds {} floats, needs {a_min}",
            item.a.len()
        );
        assert!(
            item.b.len() >= b_min,
            "batched SGEMM item {index} B holds {} floats, needs {b_min}",
            item.b.len()
        );
        assert!(
            index == 0 || item.c_offset >= previous_end,
            "batched SGEMM output windows must be ascending and disjoint: item \
             {index} starts at {} but the previous window ends at {previous_end}",
            item.c_offset
        );
        previous_end = item.c_offset + c_window;
        assert!(
            previous_end <= c.len(),
            "batched SGEMM item {index} writes up to {previous_end}, C holds {}",
            c.len()
        );
    }
    ensure_threading();
    let a_ptrs: Vec<*const f32> = items.iter().map(|i| i.a.as_ptr()).collect();
    let b_ptrs: Vec<*const f32> = items.iter().map(|i| i.b.as_ptr()).collect();
    // Sound because the loop above proved the windows are disjoint and in
    // bounds, so no two raw pointers below can alias.
    let c_base = c.as_mut_ptr();
    let c_ptrs: Vec<*mut f32> = items
        .iter()
        .map(|i| unsafe { c_base.add(i.c_offset) })
        .collect();
    unsafe {
        mlas_sgemm_batch(
            trans_a as c_int,
            trans_b as c_int,
            m,
            n,
            k,
            alpha,
            a_ptrs.as_ptr(),
            lda,
            b_ptrs.as_ptr(),
            ldb,
            beta,
            c_ptrs.as_ptr(),
            ldc,
            items.len(),
        );
    }
}

/// Blocked n-bit quantized GEMM compute type, mirroring MLAS's
/// `MLAS_QNBIT_GEMM_COMPUTE_TYPE`. These are the two float-input variants used
/// by the CPU `MatMulNBits` decode path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SQNBitComputeType {
    /// fp32 activation, fp32 accumulate (`SQNBIT_CompFp32`).
    Fp32,
    /// int8 activation, int32 accumulate (`SQNBIT_CompInt8`); ONNX
    /// `accuracy_level=4`.
    Int8,
}

impl SQNBitComputeType {
    #[inline]
    fn raw(self) -> c_int {
        // Values must match the MLAS_QNBIT_GEMM_COMPUTE_TYPE enum in
        // vendor/mlas/.../inc/mlas_qnbit.h.
        match self {
            SQNBitComputeType::Fp32 => 0, // SQNBIT_CompFp32
            SQNBitComputeType::Int8 => 3, // SQNBIT_CompInt8
        }
    }
}

/// Returns whether MLAS has a blocked n-bit GEMM kernel for the current host
/// and the given `(bits, block_len, compute_type)`. Callers must gate every
/// [`SQNBitPackedB`] / [`sqnbit_gemm`] use on this being `true`.
pub fn sqnbit_gemm_available(bits: usize, blk_len: usize, comp: SQNBitComputeType) -> bool {
    unsafe { mlas_qnbit_gemm_available(bits, blk_len, comp.raw()) != 0 }
}

/// Return the exact MLAS packed-B buffer size for a blocked n-bit weight.
///
/// The packed representation is specific to the current MLAS dispatch, compute
/// type, and host ISA. `None` means MLAS cannot consume this configuration.
pub fn sqnbit_packed_b_size(
    n: usize,
    k: usize,
    bits: usize,
    blk_len: usize,
    has_zp: bool,
    comp: SQNBitComputeType,
) -> Option<usize> {
    if !sqnbit_gemm_available(bits, blk_len, comp) {
        return None;
    }
    let size =
        unsafe { mlas_qnbit_gemm_pack_b_size(n, k, bits, blk_len, has_zp as c_int, comp.raw()) };
    (size != 0).then_some(size)
}

/// MLAS-packed blockwise-quantized B weight for [`sqnbit_gemm`], mirroring how
/// ORT pre-packs the constant `MatMulNBits` initializer once and reuses it.
///
/// The `B` bytes, scales, and optional zero points use the standard ONNX
/// `MatMulNBits` layout (`[N, ceil(K/blk_len), blk_len*bits/8]`, LSB-first
/// nibbles; scales `[N, ceil(K/blk_len)]`; packed uint8 zero points). For
/// `Fp32` compute MLAS repacks only the nibbles and consumes scales/zero points
/// at GEMM time (kept here so the packed weight is self-contained). For `Int8`
/// compute MLAS bakes scale and zero point into per-block sums inside the packed
/// buffer, so `scale`/`zp` are unused at GEMM time. A default (absent) zero
/// point is the ONNX/MLAS midpoint (8 for int4), so symmetric weights need no
/// zero point.
pub struct SQNBitPackedB {
    ptr: *mut u8,
    layout: std::alloc::Layout,
    n: usize,
    k: usize,
    bits: usize,
    blk_len: usize,
    comp: SQNBitComputeType,
    has_zp: bool,
    scale: Vec<f32>,
    zp: Option<Vec<u8>>,
}

// SAFETY: identical rationale to `PackedB`: construction fully initializes the
// packed allocation and the owned scale/zp vectors, all of which are immutable
// afterward. `sqnbit_gemm` only reads them, so sharing across threads (e.g.
// MLAS's own tile parallelism) is race-free.
unsafe impl Send for SQNBitPackedB {}
unsafe impl Sync for SQNBitPackedB {}

/// Live heap bytes currently retained by all [`SQNBitPackedB`] instances: the
/// MLAS packed allocation plus the owned scale and zero-point copies. Maintained
/// by construction/`Drop` so callers can compare the memory plan's *predicted*
/// packed footprint against the bytes the kernels *actually* hold, in the same
/// run (see the `MatMulNBits` accounting tests). This is heap only; it excludes
/// the still-mapped on-disk weights and any GEMM workspace.
static SQNBIT_PACKED_LIVE_BYTES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Snapshot of [`SQNBIT_PACKED_LIVE_BYTES`]: the heap bytes all live
/// `SQNBitPackedB` instances currently retain (packed buffer + scale copy +
/// optional zero-point copy).
pub fn sqnbit_packed_live_bytes() -> usize {
    SQNBIT_PACKED_LIVE_BYTES.load(std::sync::atomic::Ordering::Relaxed)
}

impl SQNBitPackedB {
    /// Heap bytes this instance owns: the packed allocation plus the scale and
    /// optional zero-point copies MLAS's `Fp32` path consumes at GEMM time.
    pub fn owned_heap_bytes(&self) -> usize {
        self.layout.size()
            + self.scale.capacity() * std::mem::size_of::<f32>()
            + self.zp.as_ref().map_or(0, Vec::capacity)
    }

    fn account_alloc(&self) {
        SQNBIT_PACKED_LIVE_BYTES.fetch_add(
            self.owned_heap_bytes(),
            std::sync::atomic::Ordering::Relaxed,
        );
    }

    /// Pack a blockwise-quantized B weight, returning `None` when MLAS reports
    /// no packing/kernel is available for this shape on the current host (the
    /// caller must then fall back to another path).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n: usize,
        k: usize,
        bits: usize,
        blk_len: usize,
        comp: SQNBitComputeType,
        quant_b_data: &[u8],
        scale: &[f32],
        zp: Option<&[u8]>,
    ) -> Option<Self> {
        let has_zp = zp.is_some();
        let size = sqnbit_packed_b_size(n, k, bits, blk_len, has_zp, comp)?;
        let layout = std::alloc::Layout::from_size_align(size, 64).unwrap();
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!ptr.is_null(), "SQNBit packed-B allocation failed");
        let zp_ptr = zp.map_or(std::ptr::null(), |z| z.as_ptr()) as *const c_void;
        unsafe {
            mlas_qnbit_gemm_pack_b(
                n,
                k,
                bits,
                blk_len,
                comp.raw(),
                quant_b_data.as_ptr() as *const c_void,
                ptr,
                scale.as_ptr(),
                has_zp as c_int,
                zp_ptr,
            );
        }
        let packed = Self {
            ptr,
            layout,
            n,
            k,
            bits,
            blk_len,
            comp,
            has_zp,
            scale: scale.to_vec(),
            zp: zp.map(<[u8]>::to_vec),
        };
        packed.account_alloc();
        Some(packed)
    }

    /// Reconstruct an MLAS packed weight from a previously packed buffer.
    ///
    /// The bytes must have been produced by [`SQNBitPackedB::new`] (or
    /// `MlasQNBitGemmPackQuantBData`) for the same host dispatch, dimensions,
    /// quantization parameters, and compute type. The buffer is copied into a
    /// 64-byte-aligned allocation so MLAS's internal aligned offsets remain
    /// valid.
    #[allow(clippy::too_many_arguments)]
    pub fn from_prepacked(
        n: usize,
        k: usize,
        bits: usize,
        blk_len: usize,
        comp: SQNBitComputeType,
        packed: &[u8],
        scale: &[f32],
        zp: Option<&[u8]>,
    ) -> Option<Self> {
        let has_zp = zp.is_some();
        let size = sqnbit_packed_b_size(n, k, bits, blk_len, has_zp, comp)?;
        if packed.len() != size {
            return None;
        }
        let layout = std::alloc::Layout::from_size_align(size, 64).unwrap();
        let ptr = unsafe { std::alloc::alloc(layout) };
        assert!(!ptr.is_null(), "SQNBit prepacked-B allocation failed");
        unsafe { std::ptr::copy_nonoverlapping(packed.as_ptr(), ptr, size) };
        let repacked = Self {
            ptr,
            layout,
            n,
            k,
            bits,
            blk_len,
            comp,
            has_zp,
            scale: scale.to_vec(),
            zp: zp.map(<[u8]>::to_vec),
        };
        repacked.account_alloc();
        Some(repacked)
    }

    /// Serialized bytes of this host- and compute-type-specific MLAS layout.
    pub fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.layout.size()) }
    }

    /// Logical `(k, n)` dimensions of the packed weight.
    pub fn dimensions(&self) -> (usize, usize) {
        (self.k, self.n)
    }
}

impl Drop for SQNBitPackedB {
    fn drop(&mut self) {
        SQNBIT_PACKED_LIVE_BYTES.fetch_sub(
            self.owned_heap_bytes(),
            std::sync::atomic::Ordering::Relaxed,
        );
        unsafe { std::alloc::dealloc(self.ptr, self.layout) };
    }
}

/// Return the exact MLAS scratch size for one `MlasQNBitGemmBatch` call with
/// this packed weight and row count.
///
/// This is a direct wrapper over `MlasQNBitGemmBatchWorkspaceSize(M, N, K,
/// BatchN=1, ...)`; non-zero means the caller must provide a workspace buffer to
/// `MlasQNBitGemmBatch`.
pub fn sqnbit_gemm_workspace_size(packed: &SQNBitPackedB, m: usize) -> usize {
    unsafe {
        mlas_qnbit_gemm_workspace_size(
            m,
            packed.n,
            packed.k,
            packed.bits,
            packed.blk_len,
            packed.has_zp as c_int,
            packed.comp.raw(),
        )
    }
}

/// Reusable scratch buffer for `MlasQNBitGemmBatch`.
///
/// MLAS documents that callers should allocate the byte count returned by
/// `MlasQNBitGemmBatchWorkspaceSize` and pass it to `MlasQNBitGemmBatch`.
/// `mlas-sys` over-allocates by 64 bytes because the current MLAS kernels round
/// the pointer up internally before using the scratch region.
#[derive(Default)]
pub struct SQNBitGemmWorkspace {
    buffer: Vec<u8>,
}

impl SQNBitGemmWorkspace {
    /// Create an empty workspace. It grows on first use and is then reused.
    pub fn new() -> Self {
        Self::default()
    }

    /// Ensure this workspace can satisfy a GEMM with `(packed, m)`, returning
    /// the MLAS-required byte count (excluding the internal alignment slack).
    pub fn reserve_for(&mut self, packed: &SQNBitPackedB, m: usize) -> usize {
        let size = sqnbit_gemm_workspace_size(packed, m);
        if size != 0 {
            let needed = size + 64;
            if self.buffer.len() < needed {
                self.buffer.resize(needed, 0);
            }
        }
        size
    }

    /// Currently allocated byte length, including the 64-byte alignment slack.
    pub fn allocated_len(&self) -> usize {
        self.buffer.len()
    }

    fn ptr_for(&mut self, required_size: usize) -> *mut u8 {
        if required_size == 0 {
            std::ptr::null_mut()
        } else {
            debug_assert!(self.buffer.len() >= required_size + 64);
            self.buffer.as_mut_ptr()
        }
    }
}

/// Compute `C = A * dequant(packed) + bias` for row-major `A` (`m x k`) and
/// `C` (`m x n`), reusing a pre-packed blockwise-quantized weight.
///
/// When `multithread` is true MLAS sees a non-null threadpool sentinel and
/// partitions the full-width GEMM internally across the process-global
/// [`WorkStealingThreadPool`]; otherwise it runs serially. `bias`, when present,
/// is added by MLAS itself (length `n`).
pub fn sqnbit_gemm(
    packed: &SQNBitPackedB,
    m: usize,
    a: &[f32],
    bias: Option<&[f32]>,
    c: &mut [f32],
    multithread: bool,
) {
    let n = packed.n;
    assert_eq!(c.len(), m * n, "C must be m*n");
    let mut workspace = SQNBitGemmWorkspace::new();
    sqnbit_gemm_with_workspace(packed, m, a, bias, c, &mut workspace, multithread);
}

/// Same as [`sqnbit_gemm`], but reuses caller-owned MLAS scratch across calls.
pub fn sqnbit_gemm_with_workspace(
    packed: &SQNBitPackedB,
    m: usize,
    a: &[f32],
    bias: Option<&[f32]>,
    c: &mut [f32],
    workspace: &mut SQNBitGemmWorkspace,
    multithread: bool,
) {
    let n = packed.n;
    assert_eq!(c.len(), m * n, "C must be m*n");
    // Contiguous output: leading dimension equals the packed weight's N.
    // SAFETY: `c` is `m * n` contiguous f32s, so writing `m` rows of `n`
    // columns at stride `n` stays in bounds.
    unsafe {
        sqnbit_gemm_into_with_workspace(
            packed,
            m,
            a,
            bias,
            c.as_mut_ptr(),
            n,
            workspace,
            multithread,
        )
    };
}

/// Same as [`sqnbit_gemm_with_workspace`], but keeps the older explicit handle
/// in the call signature. MLAS sees a non-null `MLAS_THREADPOOL*`, partitions the
/// QNBit batch internally, and the standalone hooks execute those partitions on
/// the process-global [`WorkStealingThreadPool`].
pub fn sqnbit_gemm_with_threadpool(
    packed: &SQNBitPackedB,
    m: usize,
    a: &[f32],
    bias: Option<&[f32]>,
    c: &mut [f32],
    workspace: &mut SQNBitGemmWorkspace,
    thread_pool: &MlasThreadPool,
) {
    let n = packed.n;
    assert_eq!(c.len(), m * n, "C must be m*n");
    let c_addr = c.as_mut_ptr() as usize;
    thread_pool.install(|| unsafe {
        sqnbit_gemm_into_with_workspace(packed, m, a, bias, c_addr as *mut f32, n, workspace, true)
    });
}

/// Compute one N-shard of `C = A * dequant(packed) + bias` into a caller-owned
/// output whose leading dimension is `ldc` (columns per row), writing this
/// shard's `packed.n` columns starting at `c` for each of the `m` rows. This
/// lets a weight partitioned along N (e.g. one shard per decode worker) write
/// its columns into a shared `[m, ldc]` output without a scatter copy; for a
/// single full-width shard `ldc == packed.n` and it matches [`sqnbit_gemm`].
///
/// # Safety
/// `c` must point at a valid f32 region covering `(m - 1) * ldc + packed.n`
/// elements (the last row needs `packed.n` columns), `ldc >= packed.n`, and no
/// other thread may write the same `[row, col]` cells concurrently.
pub unsafe fn sqnbit_gemm_into(
    packed: &SQNBitPackedB,
    m: usize,
    a: &[f32],
    bias: Option<&[f32]>,
    c: *mut f32,
    ldc: usize,
    multithread: bool,
) {
    let mut workspace = SQNBitGemmWorkspace::new();
    unsafe {
        sqnbit_gemm_into_with_workspace(packed, m, a, bias, c, ldc, &mut workspace, multithread);
    }
}

/// Same as [`sqnbit_gemm_into`], but reuses caller-owned MLAS scratch.
///
/// # Safety
/// Same requirements as [`sqnbit_gemm_into`].
#[allow(clippy::too_many_arguments)]
pub unsafe fn sqnbit_gemm_into_with_workspace(
    packed: &SQNBitPackedB,
    m: usize,
    a: &[f32],
    bias: Option<&[f32]>,
    c: *mut f32,
    ldc: usize,
    workspace: &mut SQNBitGemmWorkspace,
    multithread: bool,
) {
    let (k, n) = (packed.k, packed.n);
    assert_eq!(a.len(), m * k, "A must be m*k");
    assert!(ldc >= n, "ldc must be >= packed N");
    if let Some(bias) = bias {
        assert_eq!(bias.len(), n, "bias must be length n");
    }
    ensure_threading();

    let ws_size = workspace.reserve_for(packed, m);
    let ws_ptr = workspace.ptr_for(ws_size);

    let zp_ptr = packed.zp.as_ref().map_or(std::ptr::null(), |z| z.as_ptr()) as *const c_void;
    let bias_ptr = bias.map_or(std::ptr::null(), <[f32]>::as_ptr);

    unsafe {
        mlas_qnbit_gemm(
            m,
            n,
            k,
            packed.bits,
            packed.blk_len,
            packed.comp.raw(),
            a.as_ptr(),
            k,
            packed.ptr,
            packed.scale.as_ptr(),
            packed.has_zp as c_int,
            zp_ptr,
            bias_ptr,
            c,
            ldc,
            ws_ptr,
            multithread as c_int,
        );
    }
}

/// Same as [`sqnbit_gemm_into_with_workspace`], but keeps the older explicit
/// handle in the call signature. The actual backing pool is the persistent
/// process-global [`WorkStealingThreadPool`].
///
/// # Safety
/// Same requirements as [`sqnbit_gemm_into`].
#[allow(clippy::too_many_arguments)]
pub unsafe fn sqnbit_gemm_into_with_threadpool(
    packed: &SQNBitPackedB,
    m: usize,
    a: &[f32],
    bias: Option<&[f32]>,
    c: *mut f32,
    ldc: usize,
    workspace: &mut SQNBitGemmWorkspace,
    thread_pool: &MlasThreadPool,
) {
    let c_addr = c as usize;
    thread_pool.install(|| unsafe {
        sqnbit_gemm_into_with_workspace(
            packed,
            m,
            a,
            bias,
            c_addr as *mut f32,
            ldc,
            workspace,
            true,
        );
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_send_sync<T: Send + Sync>() {}

    /// `sgemm_batch` must produce exactly what the same items produce
    /// when issued one `sgemm` at a time. Batching only changes how MLAS
    /// partitions threads, never the arithmetic, so this is a bit-exactness
    /// assertion and not a tolerance check.
    ///
    /// The `transB` case is the one the grouped mixture-of-experts path uses:
    /// expert weights arrive as `[out_features, in_features]`, i.e. already
    /// transposed relative to what a plain `A*B` wants.
    #[test]
    fn batched_sgemm_is_bit_identical_to_serial_calls() {
        for &(m, n, k, batch) in &[
            (1usize, 96usize, 64usize, 8usize),
            (3, 32, 48, 5),
            (7, 17, 23, 2),
            (1, 1, 1, 3),
        ] {
            for &trans_b in &[false, true] {
                let ldb = if trans_b { k } else { n };
                let a: Vec<f32> = (0..batch * m * k)
                    .map(|i| ((i as f32) * 0.017).sin() * 1.3)
                    .collect();
                let banks: Vec<Vec<f32>> = (0..batch)
                    .map(|e| {
                        (0..n * k)
                            .map(|i| ((i as f32 + e as f32 * 7.0) * 0.011).cos() * 0.9)
                            .collect()
                    })
                    .collect();
                let refs: Vec<&[f32]> = banks.iter().map(Vec::as_slice).collect();

                let items: Vec<SgemmBatchItem<'_>> = refs
                    .iter()
                    .enumerate()
                    .map(|(e, b)| SgemmBatchItem {
                        a: &a[e * m * k..],
                        b,
                        c_offset: e * m * n,
                    })
                    .collect();
                let mut batched = vec![0.0f32; batch * m * n];
                sgemm_batch(
                    false,
                    trans_b,
                    m,
                    n,
                    k,
                    1.0,
                    &items,
                    k,
                    ldb,
                    0.0,
                    &mut batched,
                    n,
                );

                let mut serial = vec![0.0f32; batch * m * n];
                for (e, bank) in banks.iter().enumerate() {
                    sgemm(
                        false,
                        trans_b,
                        m,
                        n,
                        k,
                        1.0,
                        &a[e * m * k..],
                        k,
                        bank,
                        ldb,
                        0.0,
                        &mut serial[e * m * n..],
                        n,
                    );
                }
                assert_eq!(
                    batched, serial,
                    "m={m} n={n} k={k} batch={batch} trans_b={trans_b}"
                );
            }
        }
    }

    /// An empty batch is a no-op rather than a panic, so callers can hand over
    /// a routing result in which no expert was selected.
    #[test]
    fn batched_sgemm_with_no_items_is_a_no_op() {
        let _a = [1.0f32; 4];
        let mut c = [7.0f32; 4];
        sgemm_batch(false, true, 1, 4, 4, 1.0, &[], 4, 4, 0.0, &mut c, 4);
        assert_eq!(c, [7.0; 4]);
    }

    /// The durable heap a `CompFp32` int4 `SQNBitPackedB` retains is exactly the
    /// packed buffer **plus** its owned scale (and, when asymmetric, zero-point)
    /// copies -- and it is linear in `N`, so per-shard packs sum to the whole.
    ///
    /// This is the mlas-sys-level anchor for the CPU EP's packed-buffer
    /// accounting: `mlas_sqnbit_scale_zp_bytes` predicts the scale/zp overhead as
    /// `N*ceil(K/blk)*4 (+ N*ceil(ceil(K/blk)/2))`, and this test proves that
    /// prediction equals what a real packed buffer actually holds. If MLAS ever
    /// changed what a packed weight retains, this fails rather than letting the
    /// predictor silently drift below the real footprint (the under-reporting
    /// direction an admission gate must never take).
    #[test]
    fn sqnbit_fp32_owned_heap_equals_packed_plus_scale_zp() {
        let comp = SQNBitComputeType::Fp32;
        let (n, k, blk) = (4864usize, 896usize, 32usize);
        if !sqnbit_gemm_available(4, blk, comp) {
            eprintln!("SQNBit fp32 int4 unavailable on host; skipping");
            return;
        }
        let blocks = k.div_ceil(blk);
        for asymmetric in [false, true] {
            let weights: Vec<f32> = (0..n * k)
                .map(|i| ((i as f32 * 0.019 + 0.13).sin()) * 1.7)
                .collect();
            let (packed_b, scales, zps, _d) = quantize_int4(&weights, n, k, blk, asymmetric);
            let zp_slice = zps.as_deref();
            let packed_size = sqnbit_packed_b_size(n, k, 4, blk, asymmetric, comp).unwrap();
            let scale_bytes = n * blocks * std::mem::size_of::<f32>();
            let zp_bytes = zp_slice.map_or(0, <[u8]>::len);
            let expected_owned = packed_size + scale_bytes + zp_bytes;

            let whole =
                SQNBitPackedB::new(n, k, 4, blk, comp, &packed_b, &scales, zp_slice).unwrap();
            assert_eq!(
                whole.owned_heap_bytes(),
                expected_owned,
                "asymmetric={asymmetric}: owned heap must be packed + scales (+ zp)"
            );

            // The packed buffer is linear in N for Fp32, so N-row shards sum to
            // the whole -- the exact partition `build_mlas_shards` performs.
            let shard_align = 16usize;
            let target = shard_align * ((n / 8).div_ceil(shard_align)).max(1);
            let mut shard_owned = 0usize;
            let mut start = 0usize;
            while start < n {
                let len = target.min(n - start).max(1);
                let ps = &packed_b[start * blocks * (blk / 2)..(start + len) * blocks * (blk / 2)];
                let ss = &scales[start * blocks..(start + len) * blocks];
                let zs = zp_slice.map(|z| {
                    let per_row = blocks.div_ceil(2);
                    &z[start * per_row..(start + len) * per_row]
                });
                let shard = SQNBitPackedB::new(len, k, 4, blk, comp, ps, ss, zs).unwrap();
                shard_owned += shard.owned_heap_bytes();
                start += len;
            }
            assert_eq!(
                shard_owned,
                whole.owned_heap_bytes(),
                "asymmetric={asymmetric}: per-shard owned heap must sum to the whole"
            );
        }
    }

    #[test]
    fn packed_b_is_send_sync() {
        assert_send_sync::<PackedB>();
    }

    /// Naive row-major triple-loop reference: C = alpha*op(A)*op(B) + beta*C.
    #[allow(clippy::too_many_arguments)]
    fn ref_sgemm(
        trans_a: bool,
        trans_b: bool,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &[f32],
        lda: usize,
        b: &[f32],
        ldb: usize,
        beta: f32,
        c: &mut [f32],
        ldc: usize,
    ) {
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for p in 0..k {
                    let av = if trans_a {
                        a[p * lda + i]
                    } else {
                        a[i * lda + p]
                    };
                    let bv = if trans_b {
                        b[j * ldb + p]
                    } else {
                        b[p * ldb + j]
                    };
                    acc += av * bv;
                }
                let cell = &mut c[i * ldc + j];
                *cell = alpha * acc + beta * *cell;
            }
        }
    }

    fn seq(n: usize, seed: f32) -> Vec<f32> {
        // Deterministic pseudo-values in a small range to keep f32 error low.
        (0..n)
            .map(|i| ((i as f32 * 0.013 + seed).sin()) * 2.0)
            .collect()
    }

    fn assert_close(a: &[f32], b: &[f32], tol: f32, ctx: &str) {
        assert_eq!(a.len(), b.len());
        for (idx, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (x - y).abs();
            let rel = diff / (y.abs().max(1.0));
            assert!(
                diff <= tol || rel <= tol,
                "{ctx}: mismatch at {idx}: mlas={x} ref={y} diff={diff}"
            );
        }
    }

    fn sqnbit_test_multithread() -> bool {
        cfg!(target_arch = "aarch64")
    }

    fn check_nn(m: usize, n: usize, k: usize) {
        let a = seq(m * k, 0.5);
        let b = seq(k * n, 1.5);
        let mut c_mlas = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        sgemm_nn(m, n, k, &a, &b, &mut c_mlas);
        ref_sgemm(false, false, m, n, k, 1.0, &a, k, &b, n, 0.0, &mut c_ref, n);
        assert_close(&c_mlas, &c_ref, 1e-3, &format!("nn {m}x{n}x{k}"));
    }

    #[test]
    fn correctness_square() {
        check_nn(64, 64, 64);
    }

    #[test]
    fn correctness_non_square_and_non_tile_multiples() {
        // Sizes deliberately not multiples of typical 8/16 tile widths.
        check_nn(1, 1, 1);
        check_nn(3, 5, 7);
        check_nn(17, 31, 13);
        check_nn(32, 512, 512);
        check_nn(33, 65, 129);
        check_nn(100, 1, 100);
        check_nn(1, 100, 100);
    }

    #[test]
    fn correctness_alpha_beta() {
        let (m, n, k) = (23, 19, 41);
        let a = seq(m * k, 0.2);
        let b = seq(k * n, 0.7);
        let base = seq(m * n, 2.0);
        let mut c_mlas = base.clone();
        let mut c_ref = base.clone();
        sgemm(
            false,
            false,
            m,
            n,
            k,
            0.5,
            &a,
            k,
            &b,
            n,
            2.0,
            &mut c_mlas,
            n,
        );
        ref_sgemm(false, false, m, n, k, 0.5, &a, k, &b, n, 2.0, &mut c_ref, n);
        assert_close(&c_mlas, &c_ref, 1e-3, "alpha_beta");
    }

    #[test]
    fn correctness_transpose_b() {
        // B stored transposed: logical B is k x n, stored as n x k with ldb=k.
        let (m, n, k) = (12, 20, 28);
        let a = seq(m * k, 0.3);
        let b_t = seq(n * k, 0.9); // n rows of length k
        let mut c_mlas = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        sgemm(
            false,
            true,
            m,
            n,
            k,
            1.0,
            &a,
            k,
            &b_t,
            k,
            0.0,
            &mut c_mlas,
            n,
        );
        ref_sgemm(
            false, true, m, n, k, 1.0, &a, k, &b_t, k, 0.0, &mut c_ref, n,
        );
        assert_close(&c_mlas, &c_ref, 1e-3, "transpose_b");
    }

    #[test]
    fn correctness_transpose_a() {
        // A stored transposed: logical A is m x k, stored as k x m with lda=m.
        let (m, n, k) = (14, 22, 18);
        let a_t = seq(k * m, 0.4); // k rows of length m
        let b = seq(k * n, 0.6);
        let mut c_mlas = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        sgemm(
            true,
            false,
            m,
            n,
            k,
            1.0,
            &a_t,
            m,
            &b,
            n,
            0.0,
            &mut c_mlas,
            n,
        );
        ref_sgemm(
            true, false, m, n, k, 1.0, &a_t, m, &b, n, 0.0, &mut c_ref, n,
        );
        assert_close(&c_mlas, &c_ref, 1e-3, "transpose_a");
    }

    #[test]
    fn correctness_packed_b() {
        for (m, n, k) in [(32usize, 512usize, 512usize), (7, 13, 19), (1, 64, 64)] {
            let a = seq(m * k, 0.5);
            let b = seq(k * n, 1.5);
            let mut c_mlas = vec![0.0f32; m * n];
            let mut c_ref = vec![0.0f32; m * n];
            let packed = PackedB::new(n, k, &b);
            sgemm_nn_packed(m, &a, &packed, &mut c_mlas);
            ref_sgemm(false, false, m, n, k, 1.0, &a, k, &b, n, 0.0, &mut c_ref, n);
            assert_close(&c_mlas, &c_ref, 1e-3, &format!("packed {m}x{n}x{k}"));
        }
    }

    #[test]
    fn float_kernel_matches_detected_isa() {
        let id = selected_float_kernel();
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let expected = if std::arch::is_x86_feature_detected!("avx512f") {
            512
        } else if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
        {
            3
        } else if std::arch::is_x86_feature_detected!("avx") {
            1
        } else {
            -1
        };
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        let expected = 0;
        eprintln!("selected f32 GEMM kernel id = {id}; expected {expected} for host ISA");
        assert_eq!(
            id, expected,
            "MLAS f32 GEMM dispatch did not match host ISA"
        );
    }

    /// The threadpool sentinel must not change results. `MlasGemmBatch`
    /// partitions across threads, so this also guards against a partitioning
    /// bug producing torn, zeroed or doubly-written output tiles.
    #[test]
    fn sgemm_nn_is_correct_when_parallelized() {
        let (m, n, k) = (129usize, 257usize, 193usize); // deliberately odd
        let a = seq(m * k, 0.5);
        let b = seq(k * n, 1.5);
        let mut got = vec![0.0f32; m * n];
        sgemm_nn(m, n, k, &a, &b, &mut got);

        let mut want = vec![0.0f32; m * n];
        ref_sgemm(false, false, m, n, k, 1.0, &a, k, &b, n, 0.0, &mut want, n);

        // Per-element bound from the actual accumulation magnitude rather than
        // a flat relative epsilon: a flat bound scaled by |want| goes to zero
        // as an output approaches zero, so a zeroed tile covering
        // small-magnitude outputs could slip through. `sum |a*b|` is the
        // quantity f32 rounding actually accumulates over.
        //
        // The constant is the standard forward error bound for a length-`k`
        // dot product, `gamma_k = k*u / (1 - k*u)` with `u = EPSILON/2`,
        // rounded up to `k * EPSILON`. Deliberately not tighter: MLAS is free
        // to reassociate the sum (blocked accumulation, FMA contraction,
        // different vector widths per ISA), so a bound derived from one host's
        // accumulation order would be flaky elsewhere. It is still ~5 orders
        // of magnitude below the value itself, so a zeroed or torn tile is
        // caught.
        let tol_scale = k as f32 * f32::EPSILON;
        for i in 0..m {
            for j in 0..n {
                let mut sum_abs = 0.0f32;
                for p in 0..k {
                    sum_abs += (a[i * k + p] * b[p * n + j]).abs();
                }
                let tol = tol_scale * sum_abs.max(f32::MIN_POSITIVE);
                let (g, w) = (got[i * n + j], want[i * n + j]);
                assert!(
                    (g - w).abs() <= tol,
                    "mismatch at ({i},{j}): got {g}, want {w}, tol {tol} \
                     (sum|a*b| = {sum_abs})"
                );
            }
        }
    }

    /// Single-thread performance probe for the medium f32 MatMul shape
    /// (32x512x512) recorded in docs/performance/KERNEL_PERF.md. Ignored by default; run
    /// with:
    ///   cargo test -p mlas-sys --release -- --ignored --nocapture perf_sgemm_medium
    #[test]
    #[ignore = "perf probe; run explicitly with --ignored --nocapture"]
    fn perf_sgemm_medium() {
        use std::time::Instant;

        let (m, n, k) = (32usize, 512usize, 512usize);
        let a = seq(m * k, 0.5);
        let b = seq(k * n, 1.5);
        let mut c = vec![0.0f32; m * n];

        // Warm up (caches + first-call platform init/dispatch).
        for _ in 0..50 {
            sgemm_nn(m, n, k, &a, &b, &mut c);
        }

        let iters = 5000u32;
        let start = Instant::now();
        for _ in 0..iters {
            sgemm_nn(m, n, k, &a, &b, &mut c);
        }
        let elapsed = start.elapsed();
        // Prevent the loop from being optimized away.
        let checksum: f32 = c.iter().copied().sum();

        let per_us = elapsed.as_secs_f64() * 1e6 / iters as f64;
        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        let gflops = flops / (per_us * 1e3);
        eprintln!(
            "vendored-MLAS SGEMM 32x512x512 single-thread (repack B/call): {per_us:.1} us/iter \
             ({gflops:.1} GFLOP/s), checksum={checksum:.3}"
        );

        // Pre-packed B (parity with ORT's constant-weight pre-packing).
        let packed = PackedB::new(n, k, &b);
        for _ in 0..50 {
            sgemm_nn_packed(m, &a, &packed, &mut c);
        }
        let start = Instant::now();
        for _ in 0..iters {
            sgemm_nn_packed(m, &a, &packed, &mut c);
        }
        let elapsed_p = start.elapsed();
        let checksum_p: f32 = c.iter().copied().sum();
        let per_us_p = elapsed_p.as_secs_f64() * 1e6 / iters as f64;
        let gflops_p = flops / (per_us_p * 1e3);
        eprintln!(
            "vendored-MLAS SGEMM 32x512x512 single-thread (pre-packed B):   {per_us_p:.1} us/iter \
             ({gflops_p:.1} GFLOP/s), checksum={checksum_p:.3}"
        );
        eprintln!(
            "recorded baselines (docs/performance/KERNEL_PERF.md): ORT 1-thread ~131 us, SimdX86 ~285 us"
        );
    }

    /// Multi-thread scaling probe for the standalone MLAS SGEMM shim. Ignored
    /// by default; run with:
    ///   cargo test -p mlas-sys --release -- --ignored --nocapture perf_sgemm_multithread
    #[test]
    #[ignore = "perf probe; run explicitly with --ignored --nocapture"]
    fn perf_sgemm_multithread() {
        use std::time::Instant;

        let (m, n, k) = (32usize, 512usize, 512usize);
        let a = seq(m * k, 0.5);
        let b = seq(k * n, 1.5);
        let flops = 2.0 * m as f64 * n as f64 * k as f64;

        for threads in [1usize, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let (per_us, checksum) = pool.install(|| {
                let mut c = vec![0.0f32; m * n];
                for _ in 0..100 {
                    sgemm_nn(m, n, k, &a, &b, &mut c);
                }
                let iters = 5000u32;
                let start = Instant::now();
                for _ in 0..iters {
                    sgemm_nn(m, n, k, &a, &b, &mut c);
                }
                let per_us = start.elapsed().as_secs_f64() * 1e6 / iters as f64;
                (per_us, c.iter().copied().sum::<f32>())
            });
            let gflops = flops / (per_us * 1e3);
            eprintln!(
                "vendored-MLAS SGEMM 32x512x512 repack-B, {threads} thread(s): {per_us:.1} us/iter \
                 ({gflops:.1} GFLOP/s), checksum={checksum:.3}"
            );
        }
        eprintln!(
            "recorded ORT baselines (docs/performance/KERNEL_PERF.md): 1-thread ~131 us, 8-thread ~28-30 us"
        );
    }

    // ---- SQNBitGemm (blocked int4) correctness ----

    /// Quantize a row-major `N x K` f32 weight to ONNX `MatMulNBits` int4
    /// blocks, returning `(packed_b, scales, zero_points, dequantized_nk)`.
    /// `packed_b` is `[N, k_blocks, block_size/2]` LSB-first nibbles; `scales`
    /// is `[N, k_blocks]`; `zero_points` (when `asymmetric`) is packed uint8
    /// `[N, ceil(k_blocks/2)]`. `dequantized_nk` is the exact `(q-zp)*scale`
    /// oracle in the same `N x K` layout.
    fn quantize_int4(
        weights_nk: &[f32],
        n: usize,
        k: usize,
        block_size: usize,
        asymmetric: bool,
    ) -> (Vec<u8>, Vec<f32>, Option<Vec<u8>>, Vec<f32>) {
        let blocks = k.div_ceil(block_size);
        let blob = block_size / 2;
        let zp_row = blocks.div_ceil(2);
        let mut packed = vec![0u8; n * blocks * blob];
        let mut scales = vec![0.0f32; n * blocks];
        let mut zps = vec![0u8; n * zp_row];
        let mut dequant = vec![0.0f32; n * k];
        for row in 0..n {
            for block in 0..blocks {
                let start = block * block_size;
                let end = (start + block_size).min(k);
                let values = &weights_nk[row * k + start..row * k + end];
                let (scale, zp) = if asymmetric {
                    let min = values.iter().copied().fold(f32::INFINITY, f32::min);
                    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let scale = ((max - min) / 15.0).max(1e-6);
                    (scale, (-min / scale).round().clamp(0.0, 15.0) as u8)
                } else {
                    let max_abs = values.iter().map(|v| v.abs()).fold(0.0, f32::max);
                    ((max_abs / 7.0).max(1e-6), 8u8)
                };
                scales[row * blocks + block] = scale;
                if asymmetric {
                    zps[row * zp_row + block / 2] |= zp << (4 * (block % 2));
                }
                for (offset, &value) in values.iter().enumerate() {
                    let q = (value / scale + zp as f32).round().clamp(0.0, 15.0) as u8;
                    packed[(row * blocks + block) * blob + offset / 2] |= q << (4 * (offset % 2));
                    dequant[row * k + start + offset] = (q as f32 - zp as f32) * scale;
                }
            }
        }
        (packed, scales, asymmetric.then_some(zps), dequant)
    }

    /// Quantize a row-major `N x K` f32 weight to ONNX `MatMulNBits` uint8
    /// blocks, returning `(packed_b, scales, zero_points, dequantized_nk)`.
    /// `packed_b` is `[N, k_blocks, block_size]`; scales and zero points are
    /// both `[N, k_blocks]`.
    fn quantize_int8(
        weights_nk: &[f32],
        n: usize,
        k: usize,
        block_size: usize,
    ) -> (Vec<u8>, Vec<f32>, Vec<u8>, Vec<f32>) {
        let blocks = k.div_ceil(block_size);
        let mut packed = vec![0u8; n * blocks * block_size];
        let mut scales = vec![0.0f32; n * blocks];
        let mut zps = vec![0u8; n * blocks];
        let mut dequant = vec![0.0f32; n * k];
        for row in 0..n {
            for block in 0..blocks {
                let start = block * block_size;
                let end = (start + block_size).min(k);
                let values = &weights_nk[row * k + start..row * k + end];
                let min = values.iter().copied().fold(f32::INFINITY, f32::min);
                let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let scale = ((max - min) / 255.0).max(1e-6);
                let zp = (-min / scale).round().clamp(0.0, 255.0) as u8;
                scales[row * blocks + block] = scale;
                zps[row * blocks + block] = zp;
                for (offset, &value) in values.iter().enumerate() {
                    let q = (value / scale + zp as f32).round().clamp(0.0, 255.0) as u8;
                    packed[(row * blocks + block) * block_size + offset] = q;
                    dequant[row * k + start + offset] = (q as f32 - zp as f32) * scale;
                }
            }
        }
        (packed, scales, zps, dequant)
    }

    fn ref_gemm_nk(
        a: &[f32],
        w_nk: &[f32],
        m: usize,
        k: usize,
        n: usize,
        bias: Option<&[f32]>,
    ) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut acc = bias.map_or(0.0, |b| b[col]);
                for depth in 0..k {
                    acc += a[row * k + depth] * w_nk[col * k + depth];
                }
                c[row * n + col] = acc;
            }
        }
        c
    }

    fn check_sqnbit(
        comp: SQNBitComputeType,
        m: usize,
        n: usize,
        k: usize,
        block_size: usize,
        asymmetric: bool,
        with_bias: bool,
    ) {
        let weights: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.017 + 0.3).sin()).collect();
        let (packed_b, scales, zps, dequant) =
            quantize_int4(&weights, n, k, block_size, asymmetric);
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i as f32 * 0.011 + 0.7).cos()) * 0.5)
            .collect();
        let bias: Option<Vec<f32>> =
            with_bias.then(|| (0..n).map(|i| (i as f32 * 0.03).sin()).collect());

        let packed = match SQNBitPackedB::new(
            n,
            k,
            4,
            block_size,
            comp,
            &packed_b,
            &scales,
            zps.as_deref(),
        ) {
            Some(p) => p,
            None => {
                eprintln!(
                    "SQNBit int4 blk={block_size} comp={comp:?} unavailable on host; skipping"
                );
                return;
            }
        };
        let mut c = vec![0.0f32; m * n];
        sqnbit_gemm(
            &packed,
            m,
            &a,
            bias.as_deref(),
            &mut c,
            sqnbit_test_multithread(),
        );
        let expected = ref_gemm_nk(&a, &dequant, m, k, n, bias.as_deref());
        let tol = match comp {
            SQNBitComputeType::Fp32 => 2e-2,
            // CompInt8 quantizes activations and the ARM64 KleidiAI path stores
            // qsi4 scales as bf16 in the packed RHS, so it is intentionally
            // approximate compared with the f32-dequant oracle.
            SQNBitComputeType::Int8 => 8e-2,
        };
        assert_close(
            &c,
            &expected,
            tol,
            &format!(
                "sqnbit {comp:?} m{m} n{n} k{k} blk{block_size} asym{asymmetric} bias{with_bias}"
            ),
        );
    }

    fn check_sqnbit_bits8_block128_with_zp(comp: SQNBitComputeType, m: usize) {
        let (n, k, block_size) = (96usize, 256usize, 128usize);
        let weights: Vec<f32> = (0..n * k)
            .map(|i| ((i as f32 * 0.019 + 0.13).sin()) * 1.7)
            .collect();
        let (packed_b, scales, zps, dequant) = quantize_int8(&weights, n, k, block_size);
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i as f32 * 0.007 + 0.31).cos()) * 0.7)
            .collect();
        let bias: Vec<f32> = (0..n).map(|i| (i as f32 * 0.03).sin()).collect();

        let packed = SQNBitPackedB::new(n, k, 8, block_size, comp, &packed_b, &scales, Some(&zps));
        if packed.is_none() {
            eprintln!("SQNBit bits=8 blk=128 comp={comp:?} unavailable on host; skipping");
            return;
        }
        let packed = packed.unwrap();
        let mut c = vec![0.0f32; m * n];
        sqnbit_gemm(
            &packed,
            m,
            &a,
            Some(&bias),
            &mut c,
            sqnbit_test_multithread(),
        );
        let expected = ref_gemm_nk(&a, &dequant, m, k, n, Some(&bias));
        assert_close(
            &c,
            &expected,
            3e-2,
            &format!("sqnbit bits8 {comp:?} block128 zp m{m}"),
        );
    }

    fn check_sqnbit_bounded_pool_matches_single_thread(comp: SQNBitComputeType) {
        let (m, n, k, block_size) = (3usize, 96usize, 256usize, 128usize);
        let weights: Vec<f32> = (0..n * k)
            .map(|i| ((i as f32 * 0.019 + 0.13).sin()) * 1.7)
            .collect();
        let (packed_b, scales, zps, _) = quantize_int4(&weights, n, k, block_size, true);
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i as f32 * 0.007 + 0.31).cos()) * 0.7)
            .collect();
        let bias: Vec<f32> = (0..n).map(|i| (i as f32 * 0.03).sin()).collect();

        let packed = match SQNBitPackedB::new(
            n,
            k,
            4,
            block_size,
            comp,
            &packed_b,
            &scales,
            zps.as_deref(),
        ) {
            Some(p) => p,
            None => {
                eprintln!("SQNBit bounded-pool check comp={comp:?} unavailable on host; skipping");
                return;
            }
        };

        let mut single = vec![0.0f32; m * n];
        sqnbit_gemm(&packed, m, &a, Some(&bias), &mut single, false);

        let pool = MlasThreadPool::new(2).expect("MLAS thread pool handle");
        assert!(pool.thread_count() >= 1);
        let mut workspace = SQNBitGemmWorkspace::new();
        let required = sqnbit_gemm_workspace_size(&packed, m);

        let mut pooled = vec![0.0f32; m * n];
        sqnbit_gemm_with_threadpool(
            &packed,
            m,
            &a,
            Some(&bias),
            &mut pooled,
            &mut workspace,
            &pool,
        );
        assert!(
            required == 0 || workspace.allocated_len() >= required + 64,
            "workspace must retain the MLAS-required scratch plus alignment slack"
        );
        assert!(
            single
                .iter()
                .zip(&pooled)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "bounded-pool SQNBit output must be bit-identical to the single-thread path"
        );

        let allocated = workspace.allocated_len();
        let mut pooled_again = vec![0.0f32; m * n];
        sqnbit_gemm_with_threadpool(
            &packed,
            m,
            &a,
            Some(&bias),
            &mut pooled_again,
            &mut workspace,
            &pool,
        );
        assert_eq!(
            workspace.allocated_len(),
            allocated,
            "workspace should be reused without reallocating for the same shape"
        );
        assert_eq!(
            pooled.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            pooled_again.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn sqnbit_multithread_uses_work_stealing_backend() {
        let (m, n, k, block_size) = (1usize, 1024usize, 1024usize, 32usize);
        let weights: Vec<f32> = (0..n * k)
            .map(|i| ((i as f32 * 0.017 + 0.11).sin()) * 1.3)
            .collect();
        let (packed_b, scales, zps, _) = quantize_int4(&weights, n, k, block_size, true);
        let packed = match SQNBitPackedB::new(
            n,
            k,
            4,
            block_size,
            SQNBitComputeType::Int8,
            &packed_b,
            &scales,
            zps.as_deref(),
        ) {
            Some(p) => p,
            None => {
                eprintln!("SQNBit Int8 unavailable on host; skipping backend stats test");
                return;
            }
        };
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i as f32 * 0.013 + 0.29).cos()) * 0.9)
            .collect();
        let mut c = vec![0.0f32; m * n];

        reset_mlas_threading_stats();
        sqnbit_gemm(&packed, m, &a, None, &mut c, true);
        let stats = mlas_threading_stats();
        assert!(
            stats.parallel_for_calls > 0,
            "multithread=true QNBit GEMM must route through MlasStandaloneParallelFor"
        );
        assert!(
            stats.scheduled_iterations >= stats.pool_threads,
            "MLAS should schedule at least one partition per reported pool lane: {stats:?}"
        );
        assert_eq!(
            stats.serial_fallback_calls, 0,
            "work-stealing backend should be available"
        );
    }

    #[test]
    fn sqnbit_partitioning_matches_mlas_qwen3_shapes() {
        let threads = 8usize;
        let cases = [
            (1024usize, 16usize, 64usize),
            (2048, 32, 64),
            (3072, 48, 64),
            (5120, 64, 80),
            (8192, 64, 128),
        ];
        for &(n, expected_tiles, expected_stride_n) in &cases {
            let p = sqnbit_mlas_partitioning(1, n, 1024, 1, threads);
            assert_eq!(p.work_items, expected_tiles, "N={n}: {p:?}");
            assert_eq!(p.thread_count_n, expected_tiles, "N={n}: {p:?}");
            assert_eq!(p.stride_n, expected_stride_n, "N={n}: {p:?}");
            assert_eq!(p.ort_claimants, threads, "N={n}: {p:?}");
            assert_eq!(p.ort_loop_counter_shards, 8, "N={n}: {p:?}");
        }
    }

    #[test]
    fn sqnbit_packed_b_is_send_sync() {
        assert_send_sync::<SQNBitPackedB>();
    }

    #[test]
    fn sqnbit_int4_compfp32_matches_reference() {
        for &blk in &[32usize, 64, 128] {
            for &m in &[1usize, 5] {
                for &asym in &[false, true] {
                    check_sqnbit(SQNBitComputeType::Fp32, m, 96, 256, blk, asym, false);
                }
            }
        }
        check_sqnbit(SQNBitComputeType::Fp32, 4, 128, 512, 32, false, true);
    }

    #[test]
    fn sqnbit_bits4_and_bits8_block128_with_zero_points_round_trip() {
        for comp in [SQNBitComputeType::Fp32, SQNBitComputeType::Int8] {
            check_sqnbit(comp, 3, 96, 256, 128, true, true);
            check_sqnbit_bounded_pool_matches_single_thread(comp);
        }
        check_sqnbit_bits8_block128_with_zp(SQNBitComputeType::Int8, 3);
    }

    /// N-sharding parity: splitting the weight into contiguous output-column
    /// shards and running each through [`sqnbit_gemm_into`] (writing its columns
    /// into a shared `[m, n]` output at stride `n`) reproduces the full-width
    /// [`sqnbit_gemm`] result. Each output column is a GEMV over K independent of
    /// the other columns, so partitioning N cannot change the arithmetic
    /// *modulo* MLAS's own SIMD column-tiling: the fp32 kernel processes columns
    /// in fixed-width tiles, so a shard boundary that falls mid-tile can reorder
    /// a block-sum reduction and shift a result by ~1 ULP. The tolerance is a few
    /// ULP (much tighter than the `2e-2` dequant-reference tolerance), which is
    /// the invariant the ep-cpu decode path relies on when it fans a projection's
    /// N-shards across the persistent decode workers (verified byte-identical
    /// end-to-end over 128 greedy tokens on Qwen2.5-0.5B).
    #[test]
    fn sqnbit_int4_n_shards_match_full() {
        let n = 96usize;
        // Include all export block sizes and a second K/block combination. The
        // deliberately uneven N shards below remain the decode-pool analogue.
        for &(k, block_size) in &[(256usize, 32usize), (256, 64), (256, 128), (384, 64)] {
            for &m in &[1usize, 5] {
                for &asym in &[false, true] {
                    for &with_bias in &[false, true] {
                        let weights: Vec<f32> =
                            (0..n * k).map(|i| (i as f32 * 0.017 + 0.3).sin()).collect();
                        let (packed_b, scales, zps, _) =
                            quantize_int4(&weights, n, k, block_size, asym);
                        let a: Vec<f32> = (0..m * k)
                            .map(|i| ((i as f32 * 0.011 + 0.7).cos()) * 0.5)
                            .collect();
                        let bias: Option<Vec<f32>> =
                            with_bias.then(|| (0..n).map(|i| (i as f32 * 0.03).sin()).collect());

                        let full = match SQNBitPackedB::new(
                            n,
                            k,
                            4,
                            block_size,
                            SQNBitComputeType::Fp32,
                            &packed_b,
                            &scales,
                            zps.as_deref(),
                        ) {
                            Some(p) => p,
                            None => {
                                eprintln!("SQNBit blk={block_size} unavailable; skipping");
                                return;
                            }
                        };
                        let mut c_full = vec![0.0f32; m * n];
                        sqnbit_gemm(
                            &full,
                            m,
                            &a,
                            bias.as_deref(),
                            &mut c_full,
                            sqnbit_test_multithread(),
                        );

                        let blocks = k.div_ceil(block_size);
                        let blob = block_size / 2;
                        let zp_row = blocks.div_ceil(2);
                        // Deliberately uneven contiguous shards, like the decode
                        // pool's per-worker segments.
                        let shards: &[(usize, usize)] = &[(0, 17), (17, 30), (47, 1), (48, 48)];
                        // multithread=false mirrors the per-worker SPMD dispatch;
                        // multithread=true mirrors the prefill shard loop.
                        let multithread_modes: &[bool] = if sqnbit_test_multithread() {
                            &[false, true]
                        } else {
                            &[false]
                        };
                        for &mt in multithread_modes {
                            let mut c_shard = vec![0.0f32; m * n];
                            for &(start, len) in shards {
                                let pb =
                                    &packed_b[start * blocks * blob..(start + len) * blocks * blob];
                                let sc = &scales[start * blocks..(start + len) * blocks];
                                let zp = zps
                                    .as_deref()
                                    .map(|z| &z[start * zp_row..(start + len) * zp_row]);
                                let packed = SQNBitPackedB::new(
                                    len,
                                    k,
                                    4,
                                    block_size,
                                    SQNBitComputeType::Fp32,
                                    pb,
                                    sc,
                                    zp,
                                )
                                .expect("shard packs when the full weight packs");
                                let bias_shard = bias.as_deref().map(|b| &b[start..start + len]);
                                // SAFETY: shards own disjoint contiguous column ranges
                                // of the [m, n] output; `start + len <= n`.
                                unsafe {
                                    sqnbit_gemm_into(
                                        &packed,
                                        m,
                                        &a,
                                        bias_shard,
                                        c_shard.as_mut_ptr().add(start),
                                        n,
                                        mt,
                                    );
                                }
                            }
                            // A few ULP at magnitude ~60 is ~2.5e-4; 1e-3 covers the
                            // worst-case tiling reorder with margin while still being
                            // ~20x tighter than the dequant-reference tolerance.
                            assert_close(
                                &c_shard,
                                &c_full,
                                1e-3,
                                &format!(
                                    "N-sharded (multithread={mt}) vs full: \
                                     k{k} blk{block_size} m{m} asym{asym} bias{with_bias}"
                                ),
                            );
                        }
                    }
                }
            }
        }
    }

    /// Regression lock for the persistent-pool decode fix: when contiguous N
    /// shards are cut on **N-tile-aligned** boundaries, the concatenated SQNBit
    /// CompFp32 GEMV is *bit-identical* (`to_bits`) to the full-width call --
    /// MLAS processes each whole N-tile the same way regardless of how many
    /// columns the shard holds. A boundary that splits an N-tile (odd column)
    /// instead forces MLAS's narrower remainder path and drifts by >= 1 ULP.
    ///
    /// The ep-cpu decode path snaps every interior shard boundary to a multiple
    /// of 16 (`MLAS_SQNBIT_DECODE_SHARD_ALIGN`) for exactly this reason; this
    /// test is the model-free proof that alignment is load-bearing (the
    /// mid-tile split below is asserted to actually differ, so the aligned
    /// assertion cannot pass vacuously).
    #[test]
    fn sqnbit_int4_tile_aligned_shards_are_bit_exact() {
        if !cfg!(any(target_arch = "x86", target_arch = "x86_64")) {
            eprintln!("skipping x86-specific SQNBit tile-alignment regression");
            return;
        }
        // qwen3-0.6b-flavoured widths: N not a multiple of the tile, block-128.
        let n = 176usize;
        let mut any_mid_tile_drift = false;
        for &(k, block_size) in &[(256usize, 128usize), (512, 32), (256, 64)] {
            for &asym in &[false, true] {
                let weights: Vec<f32> =
                    (0..n * k).map(|i| (i as f32 * 0.017 + 0.3).sin()).collect();
                let (packed_b, scales, zps, _) = quantize_int4(&weights, n, k, block_size, asym);
                let a: Vec<f32> = (0..k)
                    .map(|i| ((i as f32 * 0.011 + 0.7).cos()) * 0.5)
                    .collect();

                let full = match SQNBitPackedB::new(
                    n,
                    k,
                    4,
                    block_size,
                    SQNBitComputeType::Fp32,
                    &packed_b,
                    &scales,
                    zps.as_deref(),
                ) {
                    Some(p) => p,
                    None => {
                        eprintln!("SQNBit blk={block_size} unavailable; skipping");
                        return;
                    }
                };
                let mut c_full = vec![0.0f32; n];
                sqnbit_gemm(&full, 1, &a, None, &mut c_full, false);

                let blocks = k.div_ceil(block_size);
                let blob = block_size / 2;
                let zp_row = blocks.div_ceil(2);
                let run_shards = |shards: &[(usize, usize)]| -> Vec<f32> {
                    let mut c = vec![0.0f32; n];
                    for &(start, len) in shards {
                        let pb = &packed_b[start * blocks * blob..(start + len) * blocks * blob];
                        let sc = &scales[start * blocks..(start + len) * blocks];
                        let zp = zps
                            .as_deref()
                            .map(|z| &z[start * zp_row..(start + len) * zp_row]);
                        let packed = SQNBitPackedB::new(
                            len,
                            k,
                            4,
                            block_size,
                            SQNBitComputeType::Fp32,
                            pb,
                            sc,
                            zp,
                        )
                        .expect("shard packs when the full weight packs");
                        // SAFETY: disjoint contiguous column ranges; start+len <= n.
                        unsafe {
                            sqnbit_gemm_into(
                                &packed,
                                1,
                                &a,
                                None,
                                c.as_mut_ptr().add(start),
                                n,
                                false,
                            );
                        }
                    }
                    c
                };

                // Tile-aligned (multiple-of-16) interior boundaries: bit-exact.
                let aligned: &[(usize, usize)] = &[(0, 16), (16, 48), (64, 64), (128, 48)];
                assert_eq!(aligned.iter().map(|&(_, l)| l).sum::<usize>(), n);
                let c_aligned = run_shards(aligned);
                let aligned_bits_match = c_aligned
                    .iter()
                    .zip(&c_full)
                    .all(|(a, b)| a.to_bits() == b.to_bits());
                assert!(
                    aligned_bits_match,
                    "k{k} blk{block_size} asym{asym}: 16-aligned N shards must be \
                     bit-identical to full-width, but differ"
                );

                // Mid-tile boundaries (odd columns split an N-tile) drift from
                // the full-width call. The drift is data-dependent, so require it
                // to appear in at least one case overall (asserted after the loop)
                // rather than every case -- enough to prove the aligned assertion
                // above is non-vacuous.
                let mid_tile: &[(usize, usize)] = &[(0, 17), (17, 30), (47, 1), (48, 128)];
                assert_eq!(mid_tile.iter().map(|&(_, l)| l).sum::<usize>(), n);
                let c_mid = run_shards(mid_tile);
                any_mid_tile_drift |= c_mid
                    .iter()
                    .zip(&c_full)
                    .any(|(a, b)| a.to_bits() != b.to_bits());
            }
        }
        assert!(
            any_mid_tile_drift,
            "expected at least one mid-tile-split N shard layout to drift from full-width \
             (non-vacuous guard); none did, so the 16-alignment fix is untested on this host"
        );
    }

    #[test]
    fn sqnbit_int4_compint8_matches_reference() {
        // Portability guard pending microsoft/onnxruntime#29853: only the AVX2
        // CompInt8 SQNBit path with M=1 and asymmetric weights is affected.
        // Keep validating AVX-512; SQNBit Int8 is not broken on all non-AVX-512 hosts.
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if !std::arch::is_x86_feature_detected!("avx512f") {
            eprintln!(
                "skipping SQNBit int4 CompInt8 reference check: AVX-512F is unavailable; \
                 AVX2 CompInt8 SQNBit M=1 asymmetric-weight bug: microsoft/onnxruntime#29853"
            );
            return;
        }
        // int8-activation compute quantizes A, so tolerances are looser.
        //
        // Cross-CPU caveat: MLAS's *AVX2* M=1 CompInt8 SQNBit microkernel with a
        // zero point (`SQ4BitGemmM1Kernel_CompInt8_avx2`, all block sizes) is
        // numerically broken -- it disagrees with the dequantized reference by
        // ~46% (mlas=6.09 vs ref=11.29), far beyond int8 quantization tolerance.
        // The AVX-512 M=1 kernel and every AVX2 M>1 kernel (which apply the zero
        // point via the precomputed block-sum correction) are correct. Verified
        // under Intel SDE (`sde64 -hsw` fails, `-skx` passes); see
        // .squad/decisions/inbox/ripley-mlas-cross-cpu.md and the upstream issue
        // draft ripley-ort-issue-draft.md. Production never hits this path: int4
        // MatMulNBits with m=1 always routes to the hand int8 decode kernel (the
        // `sqnbit_decode_min() >= 2` crossover), and `try_mlas_sqnbit` additionally
        // refuses M=1 asymmetric CompInt8 on non-AVX-512 hosts. So the M=1
        // asymmetric case only exercises an MLAS capability we deliberately avoid;
        // gate it to AVX-512 hosts (where it is correct) rather than asserting a
        // value MLAS computes wrong on AVX2.
        // MLAS installs its correct AVX-512 SQNBit dispatch only when the host
        // has AVX512F *and* the core trio BW+DQ+VL (vendored platform.cpp:572,
        // under the AVX512F check at :547); AVX512F alone falls back to the
        // Avx2/Avx2vnni dispatch, i.e. the broken kernel. Mirror that exact gate
        // so the skip condition matches production's `host_has_mlas_sqnbit_avx512`.
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let host_has_avx512 = std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512dq")
            && std::arch::is_x86_feature_detected!("avx512vl");
        for &blk in &[32usize, 64, 128] {
            for &m in &[1usize, 8] {
                for &asym in &[false, true] {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    if m == 1 && asym && !host_has_avx512 {
                        eprintln!(
                            "skipping MLAS-broken AVX2 M=1 asymmetric CompInt8 blk{blk} \
                             (production uses the hand int8 kernel here)"
                        );
                        continue;
                    }
                    check_sqnbit(SQNBitComputeType::Int8, m, 96, 256, blk, asym, false);
                }
            }
        }
        check_sqnbit(SQNBitComputeType::Int8, 4, 128, 512, 32, false, true);
    }

    /// Perf probe for int4 blockwise GEMM (decode M=1 + prefill M=32) at 1 and 8
    /// threads. Ignored by default; run with:
    ///   cargo test -p mlas-sys --release -- --ignored --nocapture perf_sqnbit
    #[test]
    #[ignore = "perf probe; run explicitly with --ignored --nocapture"]
    fn perf_sqnbit() {
        use std::time::Instant;
        for &(k, n) in &[(2048usize, 2048usize), (4096, 11008)] {
            let weights: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.017).sin()).collect();
            let (packed_b, scales, _zps, _d) = quantize_int4(&weights, n, k, 32, false);
            for comp in [SQNBitComputeType::Fp32, SQNBitComputeType::Int8] {
                let packed = match SQNBitPackedB::new(n, k, 4, 32, comp, &packed_b, &scales, None) {
                    Some(p) => p,
                    None => continue,
                };
                for &m in &[1usize, 32] {
                    let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.011).cos()).collect();
                    for threads in [1usize, 8] {
                        let pool = rayon::ThreadPoolBuilder::new()
                            .num_threads(threads)
                            .build()
                            .unwrap();
                        let per_us = pool.install(|| {
                            let mut c = vec![0.0f32; m * n];
                            for _ in 0..20 {
                                sqnbit_gemm(&packed, m, &a, None, &mut c, true);
                            }
                            let iters = 200u32;
                            let start = Instant::now();
                            for _ in 0..iters {
                                sqnbit_gemm(&packed, m, &a, None, &mut c, true);
                            }
                            start.elapsed().as_secs_f64() * 1e6 / iters as f64
                        });
                        eprintln!(
                            "SQNBit int4 {comp:?} K={k} N={n} M={m} {threads}t: {per_us:.1} us/iter"
                        );
                    }
                }
            }
        }
    }

    fn round_up(value: usize, multiple: usize) -> usize {
        value.div_ceil(multiple) * multiple
    }

    /// Naive NCHW convolution reference (group=1) with optional bias.
    #[allow(clippy::too_many_arguments)]
    fn ref_conv_nchw(
        input: &[f32],
        filter: &[f32],
        bias: Option<&[f32]>,
        n: usize,
        cin: usize,
        hin: usize,
        win: usize,
        cout: usize,
        kh: usize,
        kw: usize,
        pad: [usize; 4],
        stride: [usize; 2],
        group: usize,
    ) -> (Vec<f32>, usize, usize) {
        let hout = (hin + pad[0] + pad[2] - kh) / stride[0] + 1;
        let wout = (win + pad[1] + pad[3] - kw) / stride[1] + 1;
        let cin_g = cin / group;
        let cout_g = cout / group;
        let mut out = vec![0.0f32; n * cout * hout * wout];
        for ni in 0..n {
            for oc in 0..cout {
                let g = oc / cout_g;
                for oy in 0..hout {
                    for ox in 0..wout {
                        let mut acc = bias.map_or(0.0, |b| b[oc]);
                        for icg in 0..cin_g {
                            let ic = g * cin_g + icg;
                            for ky in 0..kh {
                                let iy = oy * stride[0] + ky;
                                if iy < pad[0] || iy - pad[0] >= hin {
                                    continue;
                                }
                                let iy = iy - pad[0];
                                for kx in 0..kw {
                                    let ix = ox * stride[1] + kx;
                                    if ix < pad[1] || ix - pad[1] >= win {
                                        continue;
                                    }
                                    let ix = ix - pad[1];
                                    let iv = input[((ni * cin + ic) * hin + iy) * win + ix];
                                    let fv = filter[(((oc * cin_g) + icg) * kh + ky) * kw + kx];
                                    acc += iv * fv;
                                }
                            }
                        }
                        out[((ni * cout + oc) * hout + oy) * wout + ox] = acc;
                    }
                }
            }
        }
        (out, hout, wout)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_nchwc_group1(
        input: &[f32],
        filter: &[f32],
        bias: Option<&[f32]>,
        n: usize,
        cin: usize,
        hin: usize,
        win: usize,
        cout: usize,
        kh: usize,
        kw: usize,
        pad: [usize; 4],
        stride: [usize; 2],
    ) -> Vec<f32> {
        let block = nchwc_block_size();
        let hout = (hin + pad[0] + pad[2] - kh) / stride[0] + 1;
        let wout = (win + pad[1] + pad[3] - kw) / stride[1] + 1;
        let nchwc_cout = round_up(cout, block);
        let filter_shape = [cout as i64, cin as i64, kh as i64, kw as i64];

        let reorder_input = cin >= block;
        let (packed_filter, conv_input, in_channels_for_shape) = if reorder_input {
            let nchwc_cin = round_up(cin, block);
            let mut pf = vec![0.0f32; nchwc_cout * nchwc_cin * kh * kw];
            nchwc_reorder_filter_bibo(&filter_shape, filter, &mut pf);
            let mut blocked = vec![0.0f32; n * nchwc_cin * hin * win];
            for ni in 0..n {
                nchwc_reorder_input_nchw(
                    &input[ni * cin * hin * win..(ni + 1) * cin * hin * win],
                    &mut blocked[ni * nchwc_cin * hin * win..(ni + 1) * nchwc_cin * hin * win],
                    cin,
                    hin * win,
                );
            }
            (pf, blocked, nchwc_cin)
        } else {
            let mut pf = vec![0.0f32; nchwc_cout * cin * kh * kw];
            nchwc_reorder_filter_bo(&filter_shape, filter, &mut pf);
            (pf, input.to_vec(), cin)
        };

        let padded_bias = bias.map(|b| {
            let mut pb = vec![0.0f32; nchwc_cout];
            pb[..cout].copy_from_slice(b);
            pb
        });

        let mut blocked_out = vec![0.0f32; n * nchwc_cout * hout * wout];
        nchwc_conv(
            &[
                n as i64,
                in_channels_for_shape as i64,
                hin as i64,
                win as i64,
            ],
            &[kh as i64, kw as i64],
            &[1, 1],
            &[pad[0] as i64, pad[1] as i64, pad[2] as i64, pad[3] as i64],
            &[stride[0] as i64, stride[1] as i64],
            &[n as i64, nchwc_cout as i64, hout as i64, wout as i64],
            1,
            &conv_input,
            &packed_filter,
            padded_bias.as_deref(),
            &mut blocked_out,
            NchwcActivation::IDENTITY,
            true,
        );

        let mut out = vec![0.0f32; n * cout * hout * wout];
        nchwc_reorder_output_nchw(
            &[n as i64, cout as i64, hout as i64, wout as i64],
            &blocked_out,
            &mut out,
        );
        out
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
    }

    fn nchwc_supported_for_tests() -> bool {
        let block = nchwc_block_size();
        if block >= 8 {
            true
        } else {
            eprintln!("skipping NCHWc test: MLAS NCHWc block size {block} is unsupported");
            false
        }
    }

    #[test]
    fn nchwc_block_size_is_supported() {
        let _ = nchwc_supported_for_tests();
    }

    #[test]
    fn nchwc_conv_pointwise_matches_reference() {
        if !nchwc_supported_for_tests() {
            return;
        }
        let block = nchwc_block_size();
        let (n, cin, hin, win, cout) = (1, 2 * block, 7, 7, 3 * block);
        let input: Vec<f32> = (0..n * cin * hin * win)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
            .collect();
        let filter: Vec<f32> = (0..cout * cin)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.05)
            .collect();
        let bias: Vec<f32> = (0..cout).map(|i| (i as f32) * 0.01).collect();
        let (want, _, _) = ref_conv_nchw(
            &input,
            &filter,
            Some(&bias),
            n,
            cin,
            hin,
            win,
            cout,
            1,
            1,
            [0; 4],
            [1, 1],
            1,
        );
        let got = run_nchwc_group1(
            &input,
            &filter,
            Some(&bias),
            n,
            cin,
            hin,
            win,
            cout,
            1,
            1,
            [0; 4],
            [1, 1],
        );
        assert!(
            max_abs_diff(&want, &got) < 1e-4,
            "diff {}",
            max_abs_diff(&want, &got)
        );
    }

    #[test]
    fn nchwc_conv_3x3_blocked_matches_reference() {
        if !nchwc_supported_for_tests() {
            return;
        }
        let block = nchwc_block_size();
        let (n, cin, hin, win, cout) = (1, block, 9, 9, block);
        let input: Vec<f32> = (0..n * cin * hin * win)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.05)
            .collect();
        let filter: Vec<f32> = (0..cout * cin * 9)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.03)
            .collect();
        let (want, _, _) = ref_conv_nchw(
            &input,
            &filter,
            None,
            n,
            cin,
            hin,
            win,
            cout,
            3,
            3,
            [1, 1, 1, 1],
            [1, 1],
            1,
        );
        let got = run_nchwc_group1(
            &input,
            &filter,
            None,
            n,
            cin,
            hin,
            win,
            cout,
            3,
            3,
            [1, 1, 1, 1],
            [1, 1],
        );
        assert!(
            max_abs_diff(&want, &got) < 1e-4,
            "diff {}",
            max_abs_diff(&want, &got)
        );
    }

    #[test]
    fn nchwc_conv_first_layer_nchw_input_matches_reference() {
        if !nchwc_supported_for_tests() {
            return;
        }
        // Input channels < block: the NCHW-input (first-layer) algorithm.
        let block = nchwc_block_size();
        let (n, cin, hin, win, cout) = (1, 3, 16, 16, block + block / 2);
        let input: Vec<f32> = (0..n * cin * hin * win)
            .map(|i| ((i % 19) as f32 - 9.0) * 0.04)
            .collect();
        let filter: Vec<f32> = (0..cout * cin * 9)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.02)
            .collect();
        let bias: Vec<f32> = (0..cout).map(|i| (i as f32) * 0.02 - 0.3).collect();
        let (want, _, _) = ref_conv_nchw(
            &input,
            &filter,
            Some(&bias),
            n,
            cin,
            hin,
            win,
            cout,
            3,
            3,
            [1, 1, 1, 1],
            [2, 2],
            1,
        );
        let got = run_nchwc_group1(
            &input,
            &filter,
            Some(&bias),
            n,
            cin,
            hin,
            win,
            cout,
            3,
            3,
            [1, 1, 1, 1],
            [2, 2],
        );
        assert!(
            max_abs_diff(&want, &got) < 1e-4,
            "diff {}",
            max_abs_diff(&want, &got)
        );
    }

    #[test]
    fn nchwc_conv_depthwise_matches_reference() {
        if !nchwc_supported_for_tests() {
            return;
        }
        // Depthwise: group == channels, one input & output channel per group.
        let block = nchwc_block_size();
        let channels = 2 * block; // must be a multiple of 4
        let (n, hin, win) = (1, 8, 8);
        let input: Vec<f32> = (0..n * channels * hin * win)
            .map(|i| ((i % 15) as f32 - 7.0) * 0.06)
            .collect();
        // Filter shape [channels, 1, 3, 3].
        let filter: Vec<f32> = (0..channels * 9)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.05)
            .collect();
        let bias: Vec<f32> = (0..channels).map(|i| (i as f32) * 0.01).collect();
        let (want, hout, wout) = ref_conv_nchw(
            &input,
            &filter,
            Some(&bias),
            n,
            channels,
            hin,
            win,
            channels,
            3,
            3,
            [1, 1, 1, 1],
            [1, 1],
            channels,
        );

        let nchwc_ch = round_up(channels, block);
        let mut pf = vec![0.0f32; nchwc_ch * 9];
        nchwc_reorder_filter_bo(&[channels as i64, 1, 3, 3], &filter, &mut pf);
        let mut blocked_in = vec![0.0f32; n * nchwc_ch * hin * win];
        nchwc_reorder_input_nchw(&input, &mut blocked_in, channels, hin * win);
        let mut padded_bias = vec![0.0f32; nchwc_ch];
        padded_bias[..channels].copy_from_slice(&bias);
        let mut blocked_out = vec![0.0f32; n * nchwc_ch * hout * wout];
        nchwc_conv(
            &[n as i64, nchwc_ch as i64, hin as i64, win as i64],
            &[3, 3],
            &[1, 1],
            &[1, 1, 1, 1],
            &[1, 1],
            &[n as i64, nchwc_ch as i64, hout as i64, wout as i64],
            nchwc_ch, // group count == blocked channel count for depthwise
            &blocked_in,
            &pf,
            Some(&padded_bias),
            &mut blocked_out,
            NchwcActivation::IDENTITY,
            true,
        );
        let mut got = vec![0.0f32; n * channels * hout * wout];
        nchwc_reorder_output_nchw(
            &[n as i64, channels as i64, hout as i64, wout as i64],
            &blocked_out,
            &mut got,
        );
        assert!(
            max_abs_diff(&want, &got) < 1e-4,
            "diff {}",
            max_abs_diff(&want, &got)
        );
    }

    #[test]
    fn nchwc_conv_relu_activation_matches_reference() {
        if !nchwc_supported_for_tests() {
            return;
        }
        let block = nchwc_block_size();
        let (n, cin, hin, win, cout) = (1, block, 5, 5, block);
        let input: Vec<f32> = (0..n * cin * hin * win)
            .map(|i| ((i % 9) as f32 - 4.0) * 0.2)
            .collect();
        let filter: Vec<f32> = (0..cout * cin)
            .map(|i| ((i % 5) as f32 - 2.0) * 0.1)
            .collect();
        let (mut want, _, _) = ref_conv_nchw(
            &input,
            &filter,
            None,
            n,
            cin,
            hin,
            win,
            cout,
            1,
            1,
            [0; 4],
            [1, 1],
            1,
        );
        for v in &mut want {
            *v = v.max(0.0);
        }
        // Reuse pointwise path but apply ReLU.
        let nchwc_cout = round_up(cout, block);
        let nchwc_cin = round_up(cin, block);
        let mut pf = vec![0.0f32; nchwc_cout * nchwc_cin];
        nchwc_reorder_filter_bibo(&[cout as i64, cin as i64, 1, 1], &filter, &mut pf);
        let mut blocked_in = vec![0.0f32; n * nchwc_cin * hin * win];
        nchwc_reorder_input_nchw(&input, &mut blocked_in, cin, hin * win);
        let mut blocked_out = vec![0.0f32; n * nchwc_cout * hin * win];
        nchwc_conv(
            &[n as i64, nchwc_cin as i64, hin as i64, win as i64],
            &[1, 1],
            &[1, 1],
            &[0; 4],
            &[1, 1],
            &[n as i64, nchwc_cout as i64, hin as i64, win as i64],
            1,
            &blocked_in,
            &pf,
            None,
            &mut blocked_out,
            NchwcActivation::RELU,
            true,
        );
        let mut got = vec![0.0f32; n * cout * hin * win];
        nchwc_reorder_output_nchw(
            &[n as i64, cout as i64, hin as i64, win as i64],
            &blocked_out,
            &mut got,
        );
        assert!(
            max_abs_diff(&want, &got) < 1e-4,
            "diff {}",
            max_abs_diff(&want, &got)
        );
    }

    #[test]
    fn nchwc_pool_max_and_average_match_reference() {
        if !nchwc_supported_for_tests() {
            return;
        }
        let block = nchwc_block_size();
        let channels = block + block / 2; // partial trailing block exercises padding
        let (n, hin, win) = (1, 8, 8);
        let (kh, kw) = (2usize, 2usize);
        let (sh, sw) = (2usize, 2usize);
        let hout = (hin - kh) / sh + 1;
        let wout = (win - kw) / sw + 1;
        let input: Vec<f32> = (0..n * channels * hin * win)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.13)
            .collect();

        let nchwc_ch = round_up(channels, block);
        let mut blocked_in = vec![0.0f32; n * nchwc_ch * hin * win];
        nchwc_reorder_input_nchw(&input, &mut blocked_in, channels, hin * win);

        for kind in [PoolKind::Maximum, PoolKind::AverageIncludePad] {
            let mut blocked_out = vec![0.0f32; n * nchwc_ch * hout * wout];
            nchwc_pool(
                kind,
                &[n as i64, nchwc_ch as i64, hin as i64, win as i64],
                &[kh as i64, kw as i64],
                &[1, 1],
                &[0, 0, 0, 0],
                &[sh as i64, sw as i64],
                &[n as i64, nchwc_ch as i64, hout as i64, wout as i64],
                &blocked_in,
                &mut blocked_out,
            );
            let mut got = vec![0.0f32; n * channels * hout * wout];
            nchwc_reorder_output_nchw(
                &[n as i64, channels as i64, hout as i64, wout as i64],
                &blocked_out,
                &mut got,
            );

            let mut want = vec![0.0f32; n * channels * hout * wout];
            for c in 0..channels {
                for oh in 0..hout {
                    for ow in 0..wout {
                        let mut acc = if matches!(kind, PoolKind::Maximum) {
                            f32::NEG_INFINITY
                        } else {
                            0.0
                        };
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let ih = oh * sh + ky;
                                let iw = ow * sw + kx;
                                let v = input[((c * hin) + ih) * win + iw];
                                if matches!(kind, PoolKind::Maximum) {
                                    acc = acc.max(v);
                                } else {
                                    acc += v;
                                }
                            }
                        }
                        if !matches!(kind, PoolKind::Maximum) {
                            acc /= (kh * kw) as f32;
                        }
                        want[((c * hout) + oh) * wout + ow] = acc;
                    }
                }
            }
            assert!(
                max_abs_diff(&want, &got) < 1e-4,
                "kind {kind:?} diff {}",
                max_abs_diff(&want, &got)
            );
        }
    }

    /// NCHW -> NCHWc -> NCHW must reproduce the original activation exactly for
    /// the kept channels, including when the channel count leaves a partial
    /// trailing block (padding lanes are added on the way in and dropped on the
    /// way out). This is the layout round-trip the graph pass relies on at
    /// region entry/exit boundaries.
    #[test]
    fn nchwc_reorder_round_trip_is_identity() {
        if !nchwc_supported_for_tests() {
            return;
        }
        let block = nchwc_block_size();
        // Exercise both an exact multiple of the block and a partial trailing
        // block (still a multiple of 4, the reorder's channel-group unit).
        for &channels in &[block, block + 4] {
            let (n, h, w) = (1usize, 5usize, 7usize);
            let plane = h * w;
            let input: Vec<f32> = (0..n * channels * plane)
                .map(|i| ((i % 17) as f32 - 8.0) * 0.07)
                .collect();

            let nchwc_ch = round_up(channels, block);
            let mut blocked = vec![7.0f32; n * nchwc_ch * plane]; // non-zero fill
            nchwc_reorder_input_nchw(&input, &mut blocked, channels, plane);

            let mut back = vec![0.0f32; n * channels * plane];
            nchwc_reorder_output_nchw(
                &[n as i64, channels as i64, h as i64, w as i64],
                &blocked,
                &mut back,
            );

            assert_eq!(back, input, "round-trip mismatch for channels={channels}");
        }
    }

    /// The standalone pool's default sizing rule must never hand a host *fewer*
    /// workers than the previous `available.clamp(1, 8)` rule did, and must
    /// scale past eight on hosts that have the cores.
    ///
    /// The old cap was inherited from the CPU EP's flat Rayon pool, whose per-op
    /// fork/join regresses past eight workers. This pool is persistent and
    /// work-stealing, so the cap only starved it: on a 32-vCPU host it left f32
    /// MatMul 1.85--2.64x slower than ORT out of the box, versus 0.92--1.30x
    /// once the pool is allowed half the machine.
    #[test]
    fn default_pool_size_is_monotone_against_the_old_eight_worker_rule() {
        for available in 1..=256usize {
            let old = available.clamp(1, 8);
            let new = resolve_default_mlas_threads(available);
            assert!(
                new >= old,
                "available={available}: new default {new} regresses below the old default {old}"
            );
            assert!(
                new <= available.max(1),
                "available={available}: new default {new} oversubscribes the host"
            );
            assert!(
                new <= MAX_MLAS_POOL_THREADS,
                "available={available}: new default {new} exceeds the pool cap"
            );
        }
        // Pin the interesting points so a future edit cannot silently drift.
        assert_eq!(resolve_default_mlas_threads(0), 1, "degenerate host");
        assert_eq!(resolve_default_mlas_threads(1), 1);
        assert_eq!(resolve_default_mlas_threads(4), 4, "small host unchanged");
        assert_eq!(resolve_default_mlas_threads(8), 8, "old ceiling unchanged");
        assert_eq!(resolve_default_mlas_threads(16), 8, "old ceiling unchanged");
        assert_eq!(
            resolve_default_mlas_threads(32),
            16,
            "half of a 32-vCPU host"
        );
        assert_eq!(resolve_default_mlas_threads(256), 64, "capped");
    }

    /// `ONNX_GENAI_CPU_DECODE_THREADS` is the CPU EP's thread-budget knob. The
    /// standalone MLAS pool used to ignore it entirely, so asking the EP for N
    /// threads left every `MlasGemmBatch` call on a differently-sized pool.
    #[allow(clippy::too_many_arguments)]
    fn qgemm_oracle(
        m: usize,
        n: usize,
        k: usize,
        a: &[u8],
        a_signed: bool,
        za: u8,
        b: &[u8],
        b_signed: bool,
        zb: &[u8],
    ) -> Vec<i32> {
        let widen = |byte: u8, signed: bool| {
            if signed {
                i32::from(byte as i8)
            } else {
                i32::from(byte)
            }
        };
        let za = widen(za, a_signed);
        let mut c = vec![0i32; m * n];
        for row in 0..m {
            for column in 0..n {
                let zb = widen(zb[column % zb.len()], b_signed);
                let mut acc = 0i32;
                for inner in 0..k {
                    acc += (widen(a[row * k + inner], a_signed) - za)
                        * (widen(b[inner * n + column], b_signed) - zb);
                }
                c[row * n + column] = acc;
            }
        }
        c
    }

    /// MLAS's integer QGEMM must reproduce the plain integer dot product for
    /// every signedness pair, including the `m == 1` decode row and shapes that
    /// are not multiples of any kernel tile.
    #[test]
    fn qgemm_i32_matches_the_integer_oracle_for_every_signedness() {
        // `u8 x i8` is the one pair a pairwise-i16 kernel can saturate; every
        // other pair is exact on every kernel we ship, on every host. Asserting
        // the exact pairs unconditionally is what makes this test a real check
        // rather than a restatement of whatever the machine happens to do.
        fn exact_here(a_signed: bool, b_signed: bool) -> bool {
            !(!a_signed && b_signed) || qgemm_u8s8_is_exact()
        }

        let mut state = 0x1234_5678_9abc_def0u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 33) as u8
        };

        for &(m, n, k) in &[
            (1, 1, 1),
            (1, 37, 61),
            (3, 8, 4),
            (5, 65, 129),
            (16, 16, 32),
        ] {
            let a: Vec<u8> = (0..m * k).map(|_| next()).collect();
            let b: Vec<u8> = (0..k * n).map(|_| next()).collect();
            for &a_signed in &[false, true] {
                for &b_signed in &[false, true] {
                    let za = next();
                    let zb = next();
                    let mut got = vec![0i32; m * n];
                    qgemm_i32(
                        m,
                        n,
                        k,
                        &a,
                        a_signed,
                        za,
                        &b,
                        b_signed,
                        QgemmZeroPoints::PerTensor(zb),
                        &mut got,
                    );
                    let want = qgemm_oracle(m, n, k, &a, a_signed, za, &b, b_signed, &[zb]);
                    if exact_here(a_signed, b_signed) {
                        assert_eq!(
                            got, want,
                            "per-tensor m={m} n={n} k={k} a_signed={a_signed} \
                             b_signed={b_signed}"
                        );
                    }

                    let zb_columns: Vec<u8> = (0..n).map(|_| next()).collect();
                    let mut got = vec![0i32; m * n];
                    qgemm_i32(
                        m,
                        n,
                        k,
                        &a,
                        a_signed,
                        za,
                        &b,
                        b_signed,
                        QgemmZeroPoints::PerColumn(&zb_columns),
                        &mut got,
                    );
                    let want = qgemm_oracle(m, n, k, &a, a_signed, za, &b, b_signed, &zb_columns);
                    if exact_here(a_signed, b_signed) {
                        assert_eq!(
                            got, want,
                            "per-column m={m} n={n} k={k} a_signed={a_signed} \
                             b_signed={b_signed}"
                        );
                    }
                }
            }
        }
    }

    /// The probe must predict what the machine actually does. A probe that
    /// always answered "exact" would silently license a lossy kernel, and one
    /// that always answered "inexact" would give up the fast path for free.
    #[test]
    fn qgemm_u8s8_probe_agrees_with_what_the_machine_actually_computes() {
        let (m, n, k) = (4, 24, 96);
        let mut state = 0x0f1e_2d3c_4b5a_6978u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 33) as u8
        };
        // Full-range operands: `u8` near 255 against `i8` near -128 is exactly
        // the pattern that overflows a pairwise `i16` accumulator.
        let a: Vec<u8> = (0..m * k).map(|_| 192 | (next() >> 2)).collect();
        let b: Vec<u8> = (0..k * n).map(|_| 0x80 | (next() >> 2)).collect();
        let mut got = vec![0i32; m * n];
        qgemm_i32(
            m,
            n,
            k,
            &a,
            false,
            0,
            &b,
            true,
            QgemmZeroPoints::PerTensor(0),
            &mut got,
        );
        let want = qgemm_oracle(m, n, k, &a, false, 0, &b, true, &[0]);
        assert_eq!(
            got == want,
            qgemm_u8s8_is_exact(),
            "the u8xi8 exactness probe disagrees with a deliberately saturating \
             workload; probe said {}",
            qgemm_u8s8_is_exact()
        );
    }

    /// A pre-packed B must give bit-identical `i32` accumulators to the
    /// unpacked path, for every signedness MLAS supports here and for both
    /// zero-point layouts.
    ///
    /// Bit-identical is the right bar: the pack only changes B's memory layout,
    /// not the arithmetic, so any difference is a packing bug rather than a
    /// tolerable reordering.
    #[test]
    fn qgemm_packed_b_is_bit_identical_to_the_unpacked_path() {
        let (m, n, k) = (5usize, 19usize, 37usize);
        let a: Vec<u8> = (0..m * k).map(|i| (i * 37 % 251) as u8).collect();
        let b: Vec<u8> = (0..k * n).map(|i| (i * 53 % 241) as u8).collect();
        let mut checked = 0usize;
        for (a_signed, b_signed) in [(false, false), (false, true), (true, true)] {
            if !a_signed && b_signed && !qgemm_u8s8_is_exact() {
                continue;
            }
            let Some(packed) = QgemmPackedB::new(n, k, &b, a_signed, b_signed) else {
                continue;
            };
            assert_eq!(packed.identity(), (k, n, a_signed, b_signed));
            let zero_point_a = 7u8;
            let per_column: Vec<u8> = (0..n).map(|i| (i * 11 % 97) as u8).collect();
            for zero_points in [
                QgemmZeroPoints::PerTensor(13),
                QgemmZeroPoints::PerColumn(&per_column),
            ] {
                let mut want = vec![0i32; m * n];
                qgemm_i32(
                    m,
                    n,
                    k,
                    &a,
                    a_signed,
                    zero_point_a,
                    &b,
                    b_signed,
                    zero_points,
                    &mut want,
                );
                let mut got = vec![0i32; m * n];
                qgemm_i32_packed(
                    m,
                    n,
                    k,
                    &a,
                    a_signed,
                    zero_point_a,
                    &packed,
                    zero_points,
                    &mut got,
                );
                assert_eq!(
                    got, want,
                    "packed and unpacked qgemm disagreed for a_signed={a_signed} \
                     b_signed={b_signed}"
                );
                checked += 1;
            }
        }
        assert!(
            checked > 0,
            "no signedness/zero-point combination was exercised, so this test proved nothing"
        );
    }

    /// The scalar requantization every caller would otherwise write by hand:
    /// round to nearest, ties to even, offset by the zero point, clamp.
    fn requantize_oracle(
        products: &[i32],
        scale: &QgemmScale<'_>,
        n: usize,
        zero_point: i32,
        signed: bool,
    ) -> Vec<u8> {
        products
            .iter()
            .enumerate()
            .map(|(index, &product)| {
                let scale = match scale {
                    QgemmScale::PerTensor(value) => *value,
                    QgemmScale::PerColumn(values) => values[index % n],
                };
                let value = ((product as f32 * scale).round_ties_even() as i64)
                    .saturating_add(i64::from(zero_point));
                if signed {
                    value.clamp(i8::MIN as i64, i8::MAX as i64) as i8 as u8
                } else {
                    value.clamp(u8::MIN as i64, u8::MAX as i64) as u8
                }
            })
            .collect()
    }

    /// The fused path must agree, byte for byte, with running the unfused
    /// `qgemm_i32` and requantizing its accumulator in scalar Rust. If it does
    /// not, a kernel that switches to it silently changes its output.
    #[test]
    fn qgemm_requantize_matches_the_unfused_accumulator_and_a_scalar_requantize() {
        // `n` deliberately straddles MLAS's 16-column block so the tail path is
        // exercised too, and `m > 1` so more than one output row is processed.
        let (m, n, k) = (3usize, 19usize, 37usize);
        let a: Vec<u8> = (0..m * k).map(|i| (i * 37 % 251) as u8).collect();
        let b: Vec<u8> = (0..k * n).map(|i| (i * 53 % 241) as u8).collect();
        let per_column_zero_points: Vec<u8> = (0..n).map(|i| (i * 11 % 97) as u8).collect();
        let per_column_scales: Vec<f32> =
            (0..n).map(|i| 1.0e-5 * (1.0 + i as f32 * 0.37)).collect();
        let mut checked = 0usize;
        for (a_signed, b_signed) in [(false, false), (false, true), (true, true)] {
            if !a_signed && b_signed && !qgemm_u8s8_is_exact() {
                continue;
            }
            let packed = QgemmPackedB::new(n, k, &b, a_signed, b_signed);
            for zero_points in [
                QgemmZeroPoints::PerTensor(13),
                QgemmZeroPoints::PerColumn(&per_column_zero_points),
            ] {
                let mut products = vec![0i32; m * n];
                qgemm_i32(
                    m,
                    n,
                    k,
                    &a,
                    a_signed,
                    7,
                    &b,
                    b_signed,
                    zero_points,
                    &mut products,
                );
                for scale in [
                    QgemmScale::PerTensor(2.0e-5),
                    QgemmScale::PerColumn(&per_column_scales),
                ] {
                    for (output_signed, output_zero_point) in [(false, 128), (true, -5)] {
                        let want = requantize_oracle(
                            &products,
                            &scale,
                            n,
                            output_zero_point,
                            output_signed,
                        );
                        for weights in [
                            Some(QgemmWeights::Dense {
                                bytes: &b,
                                signed: b_signed,
                            }),
                            packed.as_ref().map(QgemmWeights::Packed),
                        ]
                        .into_iter()
                        .flatten()
                        {
                            let mut scratch = vec![0i32; m * n];
                            let mut got = vec![0u8; m * n];
                            qgemm_requantize(
                                m,
                                n,
                                k,
                                &a,
                                a_signed,
                                7,
                                weights,
                                zero_points,
                                scale,
                                &mut got,
                                output_signed,
                                output_zero_point,
                                &mut scratch,
                            );
                            assert_eq!(
                                got, want,
                                "fused requantize disagreed with the scalar oracle for \
                                 a_signed={a_signed} b_signed={b_signed} \
                                 output_signed={output_signed}"
                            );
                            checked += 1;
                        }
                    }
                }
            }
        }
        assert!(
            checked > 0,
            "no combination was exercised, so this test proved nothing"
        );
    }

    /// Saturation is part of the contract: an accumulator that scales past the
    /// output dtype's range must clamp, not wrap.
    #[test]
    fn qgemm_requantize_saturates_instead_of_wrapping() {
        let (m, n, k) = (1usize, 8usize, 16usize);
        // Maximum-magnitude activations against maximum-magnitude weights, with
        // a scale large enough that every column overflows the byte range.
        let a = vec![255u8; m * k];
        let b = vec![255u8; k * n];
        let mut scratch = vec![0i32; m * n];
        let mut got = vec![0u8; m * n];
        qgemm_requantize(
            m,
            n,
            k,
            &a,
            false,
            0,
            QgemmWeights::Dense {
                bytes: &b,
                signed: false,
            },
            QgemmZeroPoints::PerTensor(0),
            QgemmScale::PerTensor(1.0e6),
            &mut got,
            false,
            0,
            &mut scratch,
        );
        assert_eq!(
            got,
            vec![255u8; m * n],
            "positive overflow must clamp to 255"
        );

        let mut got_signed = vec![0u8; m * n];
        qgemm_requantize(
            m,
            n,
            k,
            &a,
            false,
            0,
            QgemmWeights::Dense {
                bytes: &b,
                signed: false,
            },
            QgemmZeroPoints::PerTensor(0),
            QgemmScale::PerTensor(-1.0e6),
            &mut got_signed,
            true,
            0,
            &mut scratch,
        );
        assert_eq!(
            got_signed,
            vec![(-128i8) as u8; m * n],
            "negative overflow must clamp to -128"
        );
    }

    #[test]
    fn qgemm_packed_b_declines_empty_shapes() {
        assert!(QgemmPackedB::new(0, 4, &[], false, false).is_none());
        assert!(QgemmPackedB::new(4, 0, &[], false, false).is_none());
    }

    #[test]
    fn qgemm_packed_b_is_send_sync() {
        assert_send_sync::<QgemmPackedB>();
    }

    #[test]
    fn qgemm_i32_tolerates_empty_shapes() {
        let mut c: Vec<i32> = Vec::new();
        qgemm_i32(
            0,
            4,
            4,
            &[],
            false,
            0,
            &[0; 16],
            false,
            QgemmZeroPoints::PerTensor(0),
            &mut c,
        );
        qgemm_i32(
            4,
            0,
            4,
            &[0; 16],
            false,
            0,
            &[],
            false,
            QgemmZeroPoints::PerTensor(0),
            &mut c,
        );
    }

    #[test]
    fn parse_thread_count_parses_and_clamps() {
        // Deliberately tests the pure parser rather than `thread_count_env`:
        // `std::env::set_var` races concurrent `getenv` in sibling tests (cargo
        // runs lib tests multi-threaded), which is a data race even for
        // distinct keys. The only thing `thread_count_env` adds is the
        // `std::env::var` lookup.
        assert_eq!(parse_thread_count("12"), Some(12));
        assert_eq!(
            parse_thread_count("  7  "),
            Some(7),
            "surrounding whitespace must be tolerated"
        );
        assert_eq!(
            parse_thread_count("9999"),
            Some(MAX_MLAS_POOL_THREADS),
            "an absurd request must clamp, not oversubscribe"
        );
        for bad in ["0", "", "abc", "-4", "3.5", "18446744073709551616"] {
            assert_eq!(
                parse_thread_count(bad),
                None,
                "{bad:?} must fall through to the next precedence level"
            );
        }
    }

    #[test]
    fn pool_thread_budget_round_trips_and_rejects_zero() {
        // Serialized against the resolver test below by exercising only the
        // atomic, which is process-local state this test fully owns and
        // restores.
        let previous = POOL_THREAD_BUDGET.load(Ordering::Acquire);
        assert_eq!(
            set_pool_thread_budget(Some(0)),
            Err("MLAS pool thread budget must be greater than zero"),
            "zero is the CPU EP's opt-out sentinel, not a legal pool size"
        );

        set_pool_thread_budget(Some(6)).expect("6 is a legal budget");
        assert_eq!(pool_thread_budget(), Some(6));

        set_pool_thread_budget(Some(9999)).expect("an absurd budget clamps");
        assert_eq!(
            pool_thread_budget(),
            Some(MAX_MLAS_POOL_THREADS),
            "the programmatic path must clamp exactly like the env path"
        );

        set_pool_thread_budget(None).expect("clearing is legal");
        assert_eq!(
            pool_thread_budget(),
            None,
            "a cleared budget must fall through to the next precedence level"
        );
        POOL_THREAD_BUDGET.store(previous, Ordering::Release);
    }
}
