//! Thin FFI wrapper around a vendored subset of ONNX Runtime's MLAS
//! single-precision GEMM (`MlasGemmBatch`).
//!
//! The vendored MLAS is compiled in its standalone `BUILD_MLAS_NO_ONNXRUNTIME`
//! mode, whose threading primitives normally serialize. This crate installs a
//! Rayon-backed parallel-for backend (see [`ensure_threading`] and
//! `vendor/shim.cpp`) so MLAS keeps its own cache-aware GEMM tile partitioning
//! while executing the tiles across the current Rayon pool — the same pool the
//! rest of `onnx-runtime-ep-cpu` uses, so there is no oversubscription. See
//! `docs/MLAS_SYS_SPIKE.md` for the original single-thread feasibility spike.

use std::os::raw::c_int;
use std::os::raw::c_void;
use std::ptr::NonNull;
use std::sync::Once;

use rayon::prelude::*;

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

/// Rayon-backed parallel-for. Runs on whatever pool is current at call time
/// (i.e. the ep-cpu global pool, or a `ThreadPool::install` scope), so MLAS
/// never spawns a second pool that would oversubscribe the machine.
unsafe extern "C" fn rayon_parallel_for(
    _rust_ctx: *mut c_void,
    iterations: isize,
    task: MlasTaskFn,
    task_ctx: *mut c_void,
) {
    if iterations <= 0 {
        return;
    }
    // Carry the opaque C++ closure pointer across Rayon worker threads as an
    // address (usize is Send + Sync). MLAS only *reads* the closure
    // (`std::function::operator() const`) and each `tid` writes a disjoint
    // output partition, so concurrent invocation is race-free.
    let task_ctx = task_ctx as usize;
    (0..iterations).into_par_iter().for_each(|tid| {
        // SAFETY: `task_ctx` is valid for the whole `MlasGemmBatch` call that
        // drives this parallel-for; each `tid` touches a disjoint output range.
        unsafe { task(task_ctx as *mut c_void, tid) };
    });
}

/// Report Rayon's current degree of parallelism to MLAS's partitioner, so the
/// GEMM is split into as many tiles as there are worker threads available.
unsafe extern "C" fn rayon_max_threads(_rust_ctx: *mut c_void) -> c_int {
    rayon::current_num_threads().max(1) as c_int
}

static THREADING_INIT: Once = Once::new();

/// Install the Rayon-backed threading backend into the vendored MLAS build.
/// Idempotent; called before every GEMM entry point. Until this runs (e.g. in
/// the mlas-sys unit tests that call the FFI directly) MLAS stays single
/// threaded, matching the original spike behaviour.
fn ensure_threading() {
    THREADING_INIT.call_once(|| unsafe {
        mlas_set_threading(rayon_parallel_for, rayon_max_threads, std::ptr::null_mut());
    });
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

/// Pre-packed B weight buffer, mirroring how ORT pre-packs constant MatMul
/// weights once and reuses the packed panel across calls.
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

/// Blocked n-bit quantized GEMM compute type, mirroring MLAS's
/// `MLAS_QNBIT_GEMM_COMPUTE_TYPE`. Only the two x86 float-input variants used
/// by the CPU `MatMulNBits` decode path are exposed.
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

impl SQNBitPackedB {
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
        if !sqnbit_gemm_available(bits, blk_len, comp) {
            return None;
        }
        let has_zp = zp.is_some();
        let size = unsafe {
            mlas_qnbit_gemm_pack_b_size(n, k, bits, blk_len, has_zp as c_int, comp.raw())
        };
        if size == 0 {
            return None;
        }
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
        Some(Self {
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
        })
    }

    /// Logical `(k, n)` dimensions of the packed weight.
    pub fn dimensions(&self) -> (usize, usize) {
        (self.k, self.n)
    }
}

impl Drop for SQNBitPackedB {
    fn drop(&mut self) {
        unsafe { std::alloc::dealloc(self.ptr, self.layout) };
    }
}

/// Compute `C = A * dequant(packed) + bias` for row-major `A` (`m x k`) and
/// `C` (`m x n`), reusing a pre-packed blockwise-quantized weight.
///
/// When `multithread` is true MLAS partitions the GEMM across the current Rayon
/// pool (see [`PackedB`] threading notes); otherwise it runs serially. `bias`,
/// when present, is added by MLAS itself (length `n`).
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
    // Contiguous output: leading dimension equals the packed weight's N.
    // SAFETY: `c` is `m * n` contiguous f32s, so writing `m` rows of `n`
    // columns at stride `n` stays in bounds.
    unsafe { sqnbit_gemm_into(packed, m, a, bias, c.as_mut_ptr(), n, multithread) };
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
    let (k, n) = (packed.k, packed.n);
    assert_eq!(a.len(), m * k, "A must be m*k");
    assert!(ldc >= n, "ldc must be >= packed N");
    if let Some(bias) = bias {
        assert_eq!(bias.len(), n, "bias must be length n");
    }
    ensure_threading();

    let ws_size = unsafe {
        mlas_qnbit_gemm_workspace_size(
            m,
            n,
            k,
            packed.bits,
            packed.blk_len,
            packed.has_zp as c_int,
            packed.comp.raw(),
        )
    };
    // MLAS rounds the workspace pointer up to an internal alignment, so
    // over-allocate to keep the aligned [start, start+ws_size) region in bounds.
    let mut workspace: Vec<u8> = if ws_size == 0 {
        Vec::new()
    } else {
        vec![0u8; ws_size + 64]
    };
    let ws_ptr = if ws_size == 0 {
        std::ptr::null_mut()
    } else {
        workspace.as_mut_ptr()
    };

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

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_send_sync<T: Send + Sync>() {}

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
    fn avx512_kernel_is_selected() {
        // Proves parity-by-construction: on this AVX-512 host MLAS's runtime
        // dispatch must pick the AVX-512F SGEMM microkernel.
        let id = selected_float_kernel();
        eprintln!("selected f32 GEMM kernel id = {id} (512 = AVX-512F)");
        assert_eq!(id, 512, "expected AVX-512F SGEMM kernel to be selected");
    }

    /// Single-thread performance probe for the medium f32 MatMul shape
    /// (32x512x512) recorded in docs/KERNEL_PERF.md. Ignored by default; run
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
            "recorded baselines (docs/KERNEL_PERF.md): ORT 1-thread ~131 us, SimdX86 ~285 us"
        );
    }

    /// Multi-thread scaling probe: measures the same 32x512x512 shape at 1 and
    /// 8 Rayon threads to confirm MLAS's own tile partitioning now runs across
    /// the pool. Ignored by default; run with:
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
            "recorded ORT baselines (docs/KERNEL_PERF.md): 1-thread ~131 us, 8-thread ~28-30 us"
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
        sqnbit_gemm(&packed, m, &a, bias.as_deref(), &mut c, true);
        let expected = ref_gemm_nk(&a, &dequant, m, k, n, bias.as_deref());
        assert_close(
            &c,
            &expected,
            2e-2,
            &format!(
                "sqnbit {comp:?} m{m} n{n} k{k} blk{block_size} asym{asymmetric} bias{with_bias}"
            ),
        );
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
                        sqnbit_gemm(&full, m, &a, bias.as_deref(), &mut c_full, true);

                        let blocks = k.div_ceil(block_size);
                        let blob = block_size / 2;
                        let zp_row = blocks.div_ceil(2);
                        // Deliberately uneven contiguous shards, like the decode
                        // pool's per-worker segments.
                        let shards: &[(usize, usize)] = &[(0, 17), (17, 30), (47, 1), (48, 48)];
                        // multithread=false mirrors the per-worker SPMD dispatch;
                        // multithread=true mirrors the prefill shard loop.
                        for &mt in &[false, true] {
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

    #[test]
    fn sqnbit_int4_compint8_matches_reference() {
        // int8-activation compute quantizes A, so tolerances are looser.
        for &blk in &[32usize, 64, 128] {
            for &m in &[1usize, 8] {
                for &asym in &[false, true] {
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
}
