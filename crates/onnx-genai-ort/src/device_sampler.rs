//! Device-side token sampling for captured decode.
//!
//! In the captured decode loop the model's `logits [1, 1, vocab]` output is the
//! only tensor we need to reduce to a single winning token id. When the logits
//! buffer is CPU-allocated, ORT copies the entire vocabulary (e.g. 151,936 f16 =
//! ~300 KiB) host-side every token inside `RunWithBinding`, and we then sample on
//! the host. onnxruntime-genai avoids this by keeping logits on the GPU and
//! reducing them with custom CUDA kernels; this module does the same.
//! [`DeviceSampler`] is the extension point for compute backends;
//! [`CudaSampler`] provides the CUDA implementation.
//!
//! The sampler applies the device-portable pipeline — temperature, top-k, top-p,
//! min-p, then greedy (argmax) or categorical selection — entirely on the device
//! pointer, copying back only the 4-byte token id(s). History-dependent
//! processors (repetition/frequency/presence penalties, grammar constraints,
//! stop sequences) and logprobs remain host-side; [`DeviceSampler::copy_row_to_host`]
//! serves those by copying the full row on demand.
//!
//! ## Context sharing
//!
//! [`cudarc::driver::CudaContext::new`] *retains the primary context* of the
//! device (`cuDevicePrimaryCtxRetain`), which is the very context ORT's built-in
//! CUDA EP drives. Device pointers are therefore valid across both, so the
//! kernels can read ORT's logits allocation directly with no cross-context copy
//! (unlike `OrtApi::CopyTensors`, which has no data-transfer path for the
//! built-in CUDA EP).
//!
//! ## Correctness
//!
//! The greedy (argmax) kernel matches the host reference argmax used elsewhere in
//! the crate: maximum value, **NaNs ignored**, **lowest index wins ties**, and
//! index `0` for an all-NaN (or empty) row. f16/bf16 are decoded to f32 with pure
//! integer bit math so the NVRTC source needs no `<cuda_fp16.h>` (keeping it
//! self-contained and header-free).

use std::sync::Mutex;

use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg};

use crate::error::{OrtError, Result};
use crate::value::DataType;

use crate::decode::DeviceSampleParams;

/// Device-side token selection over logits that remain in device memory.
///
/// Compute backends implement this interface to reduce `[rows, vocab]` logits to
/// one token id per row — applying temperature/top-k/top-p/min-p and the final
/// greedy or categorical pick — without copying the full vocabulary to the host.
pub(crate) trait DeviceSampler: Send {
    /// Select one token id per row from the device logits buffer at `ptr_addr`,
    /// applying `params` on-device.
    fn sample(
        &self,
        dtype: DataType,
        ptr_addr: usize,
        rows: usize,
        vocab: usize,
        params: &DeviceSampleParams,
    ) -> Result<Vec<u32>>;

    fn copy_row_to_host(
        &self,
        dtype: DataType,
        ptr_addr: usize,
        len: usize,
        dst: &mut [u8],
    ) -> Result<()>;

    fn name(&self) -> &str;
}

/// Threads per block. One block reduces the whole row via a grid-stride loop
/// then a shared-memory tree reduction, so this is also the shared-array width.
const BLOCK: u32 = 1024;

/// Whether to issue a device-wide `cuCtxSynchronize` before reading the logits.
/// ORT's built-in CUDA EP synchronizes its compute stream at the end of each
/// `Run`, so by the time we regain control the logits are already visible and
/// this wait returns immediately (measured: no per-token cost). It is kept on by
/// default as a correctness guard against any ORT configuration that leaves the
/// stream running asynchronously; set `ONNX_GENAI_ARGMAX_CTX_SYNC=0` to drop it.
fn ctx_sync_enabled() -> bool {
    std::env::var("ONNX_GENAI_ARGMAX_CTX_SYNC")
        .map(|v| v != "0" && !v.is_empty())
        .unwrap_or(true)
}

/// NVRTC source: three argmax entry points (f16 / bf16 / f32). Each launches
/// **one block per row** (`blockIdx.x` selects the row) and reduces that row's
/// `vocab` contiguous elements to the index of its maximum, ignoring NaN and
/// breaking ties toward the lowest index, writing index 0 when nothing is valid.
/// Handling `rows > 1` lets speculative decoding argmax every verified position
/// (`logits [1, N, vocab]`) in a single launch, not just the final token.
const ARGMAX_SRC: &str = r#"
#define BLOCK 1024

// binary16 -> f32 via pure integer bit math (no <cuda_fp16.h> dependency).
__device__ __forceinline__ float h2f(unsigned short h) {
    unsigned int s = ((unsigned int)(h & 0x8000u)) << 16;
    unsigned int e = (h >> 10) & 0x1Fu;
    unsigned int m = h & 0x3FFu;
    unsigned int out;
    if (e == 0u) {
        if (m == 0u) {
            out = s;
        } else {
            e = 1u;
            while ((m & 0x400u) == 0u) { m <<= 1; e--; }
            m &= 0x3FFu;
            out = s | ((e + 112u) << 23) | (m << 13);
        }
    } else if (e == 0x1Fu) {
        out = s | 0x7F800000u | (m << 13);
    } else {
        out = s | ((e + 112u) << 23) | (m << 13);
    }
    return __int_as_float((int)out);
}

// bfloat16 -> f32: high 16 bits of the f32.
__device__ __forceinline__ float bf2f(unsigned short h) {
    return __int_as_float((int)(((unsigned int)h) << 16));
}

__device__ __forceinline__ void block_reduce_argmax(float best, int bidx, int* out, int row) {
    __shared__ float sval[BLOCK];
    __shared__ int   sidx[BLOCK];
    int tid = threadIdx.x;
    sval[tid] = best;
    sidx[tid] = bidx;
    __syncthreads();
    for (int off = BLOCK >> 1; off > 0; off >>= 1) {
        if (tid < off) {
            float ov = sval[tid + off];
            int   oi = sidx[tid + off];
            float cv = sval[tid];
            int   ci = sidx[tid];
            if (ov > cv || (ov == cv && oi < ci)) {
                sval[tid] = ov;
                sidx[tid] = oi;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[row] = (sidx[0] == 0x7fffffff) ? 0 : sidx[0];
    }
}

extern "C" __global__ void argmax_f16(const unsigned short* x, int rows, int vocab, int* out) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const unsigned short* r = x + (size_t)row * (size_t)vocab;
    const float NEG_INF = __int_as_float(0xff800000);
    float best = NEG_INF;
    int   bidx = 0x7fffffff;
    for (int i = threadIdx.x; i < vocab; i += BLOCK) {
        float v = h2f(r[i]);
        if (v == v && (v > best || (v == best && i < bidx))) { best = v; bidx = i; }
    }
    block_reduce_argmax(best, bidx, out, row);
}

extern "C" __global__ void argmax_bf16(const unsigned short* x, int rows, int vocab, int* out) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const unsigned short* r = x + (size_t)row * (size_t)vocab;
    const float NEG_INF = __int_as_float(0xff800000);
    float best = NEG_INF;
    int   bidx = 0x7fffffff;
    for (int i = threadIdx.x; i < vocab; i += BLOCK) {
        float v = bf2f(r[i]);
        if (v == v && (v > best || (v == best && i < bidx))) { best = v; bidx = i; }
    }
    block_reduce_argmax(best, bidx, out, row);
}

extern "C" __global__ void argmax_f32(const float* x, int rows, int vocab, int* out) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float* r = x + (size_t)row * (size_t)vocab;
    const float NEG_INF = __int_as_float(0xff800000);
    float best = NEG_INF;
    int   bidx = 0x7fffffff;
    for (int i = threadIdx.x; i < vocab; i += BLOCK) {
        float v = r[i];
        if (v == v && (v > best || (v == best && i < bidx))) { best = v; bidx = i; }
    }
    block_reduce_argmax(best, bidx, out, row);
}
"#;

/// NVRTC source: the non-greedy sampling pipeline (f16 / bf16 / f32), one block
/// per row (`blockIdx.x` selects the row). Each block runs the full device
/// pipeline over its `vocab` contiguous logits — temperature scaling, numerically
/// stable softmax, top-k, top-p (nucleus), min-p, then an inverse-CDF categorical
/// draw — writing the selected token id into `out[row]`. The stages are composed
/// from modular `__device__` helpers but issued as a single fixed launch per
/// dtype so the sequence stays stable for later CUDA-graph capture.
///
/// Every threshold is a lower bound on probability, so a token survives iff its
/// probability is `>= max(topk_thresh, topp_thresh, minp_thresh)`. The argmax
/// token always has the maximum probability, so it always survives every filter
/// (matching the greedy invariant), which also guarantees the survivor set is
/// non-empty. f16/bf16 are decoded with pure integer bit math (no
/// `<cuda_fp16.h>`), matching the argmax kernels. The RNG value is applied per
/// row (each row samples independently with the same `rng`; see the Rust doc on
/// multi-row handling).
const SAMPLE_SRC: &str = r#"
#define BLOCK 1024

__device__ __forceinline__ float h2f(unsigned short h) {
    unsigned int s = ((unsigned int)(h & 0x8000u)) << 16;
    unsigned int e = (h >> 10) & 0x1Fu;
    unsigned int m = h & 0x3FFu;
    unsigned int out;
    if (e == 0u) {
        if (m == 0u) {
            out = s;
        } else {
            e = 1u;
            while ((m & 0x400u) == 0u) { m <<= 1; e--; }
            m &= 0x3FFu;
            out = s | ((e + 112u) << 23) | (m << 13);
        }
    } else if (e == 0x1Fu) {
        out = s | 0x7F800000u | (m << 13);
    } else {
        out = s | ((e + 112u) << 23) | (m << 13);
    }
    return __int_as_float((int)out);
}

__device__ __forceinline__ float bf2f(unsigned short h) {
    return __int_as_float((int)(((unsigned int)h) << 16));
}

// Block-wide max reduction; returns the max to every thread (broadcast via s[0]).
__device__ __forceinline__ float blk_max(float v) {
    __shared__ float s[BLOCK];
    int t = threadIdx.x;
    s[t] = v;
    __syncthreads();
    for (int off = BLOCK >> 1; off > 0; off >>= 1) {
        if (t < off) { float o = s[t + off]; if (o > s[t]) s[t] = o; }
        __syncthreads();
    }
    float r = s[0];
    __syncthreads();
    return r;
}

// Block-wide sum reduction; returns the sum to every thread.
__device__ __forceinline__ float blk_sum(float v) {
    __shared__ float s[BLOCK];
    int t = threadIdx.x;
    s[t] = v;
    __syncthreads();
    for (int off = BLOCK >> 1; off > 0; off >>= 1) {
        if (t < off) { s[t] += s[t + off]; }
        __syncthreads();
    }
    float r = s[0];
    __syncthreads();
    return r;
}

// `w` holds temperature-scaled logits (NaN entries pre-set to -inf) and `m` is
// their max. Turns `w` into a filtered, renormalizable probability row and
// writes the inverse-CDF categorical pick into out[row].
__device__ void finish_row(float* w, int vocab, float m,
                           int top_k, float top_p, float min_p, float rng,
                           int* out, int row) {
    int tid = threadIdx.x;
    const float NEG_INF = __int_as_float(0xff800000);
    const float POS_INF = __int_as_float(0x7f800000);
    // All-NaN / empty row: match the argmax convention of index 0.
    if (m == NEG_INF) { if (tid == 0) out[row] = 0; return; }

    // Stable softmax: exp(z - m), summed across the block.
    float ls = 0.0f;
    for (int i = tid; i < vocab; i += BLOCK) {
        float z = w[i];
        float e = (z == NEG_INF) ? 0.0f : expf(z - m);
        w[i] = e;
        ls += e;
    }
    float S = blk_sum(ls);
    if (!(S > 0.0f)) { if (tid == 0) out[row] = 0; return; }
    float invS = 1.0f / S;
    for (int i = tid; i < vocab; i += BLOCK) { w[i] *= invS; }
    __syncthreads();
    // The max logit maps to exp(0)=1, so the max probability is 1/S.
    float p_max = invS;

    // min-p: keep prob >= min_p * p_max.
    float minp_thresh = (min_p > 0.0f) ? (min_p * p_max) : 0.0f;

    // top-k: threshold = k-th largest probability (iterative selection of the
    // next-highest value strictly below the running threshold; exact for
    // distinct probabilities). O(k) block passes over the row.
    float topk_thresh = 0.0f;
    if (top_k > 0 && top_k < vocab) {
        float thr = POS_INF;
        for (int it = 0; it < top_k; ++it) {
            float lm = NEG_INF;
            for (int i = tid; i < vocab; i += BLOCK) {
                float p = w[i];
                if (p < thr && p > lm) lm = p;
            }
            float cur = blk_max(lm);
            thr = cur;
            if (thr == NEG_INF) break;
        }
        topk_thresh = (thr == NEG_INF) ? 0.0f : thr;
    }

    // top-p (nucleus): threshold = sup{ t : sum(prob >= t) >= top_p }, found by
    // binary search on the cumulative mass (no full sort needed). Keeping
    // prob >= this threshold reproduces the nucleus for distinct probabilities.
    float topp_thresh = 0.0f;
    if (top_p > 0.0f && top_p < 1.0f) {
        float lo = 0.0f, hi = p_max;
        // 32 bisections exhaust f32 precision on [0, p_max]; more only burns
        // full-vocab reductions without moving the threshold.
        for (int it = 0; it < 32; ++it) {
            float mid = 0.5f * (lo + hi);
            float loc = 0.0f;
            for (int i = tid; i < vocab; i += BLOCK) {
                float p = w[i];
                if (p >= mid) loc += p;
            }
            float sm = blk_sum(loc);
            if (sm >= top_p) lo = mid; else hi = mid;
        }
        topp_thresh = lo;
    }

    // A token survives iff its prob >= the strongest (largest) lower bound.
    float T = topk_thresh;
    if (topp_thresh > T) T = topp_thresh;
    if (minp_thresh > T) T = minp_thresh;

    // Renormalizing mass of the survivor set.
    float loc = 0.0f;
    for (int i = tid; i < vocab; i += BLOCK) {
        float p = w[i];
        if (p >= T) loc += p;
    }
    float Ssurv = blk_sum(loc);

    // Inverse-CDF draw in index order over the survivors (deterministic and
    // trivially reproducible on the host reference oracle).
    if (tid == 0) {
        float target = rng * Ssurv;
        float acc = 0.0f;
        int chosen = -1;
        int last_survivor = -1;
        for (int i = 0; i < vocab; i++) {
            float p = w[i];
            if (p >= T) {
                last_survivor = i;
                acc += p;
                if (acc > target) { chosen = i; break; }
            }
        }
        // Float non-associativity: the sequential `acc` can fall just short of
        // the parallel `Ssurv`, so an extreme-tail `target` may leave `chosen`
        // unset. Fall back to the LAST survivor (mirrors the host oracle), never
        // token 0 which may have been filtered out. `last_survivor` is always
        // >= 0 because the argmax survives any valid threshold.
        out[row] = (chosen >= 0) ? chosen : (last_survivor >= 0 ? last_survivor : 0);
    }
}

extern "C" __global__ void sample_f16(const unsigned short* x, int rows, int vocab,
                                       float* work, float temperature, int top_k,
                                       float top_p, float min_p, float rng, int* out) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const unsigned short* r = x + (size_t)row * (size_t)vocab;
    float* w = work + (size_t)row * (size_t)vocab;
    const float NEG_INF = __int_as_float(0xff800000);
    float inv_t = (temperature > 0.0f && temperature != 1.0f) ? (1.0f / temperature) : 1.0f;
    float lmax = NEG_INF;
    for (int i = threadIdx.x; i < vocab; i += BLOCK) {
        float v = h2f(r[i]);
        if (v == v) { v *= inv_t; w[i] = v; if (v > lmax) lmax = v; }
        else { w[i] = NEG_INF; }
    }
    float m = blk_max(lmax);
    finish_row(w, vocab, m, top_k, top_p, min_p, rng, out, row);
}

extern "C" __global__ void sample_bf16(const unsigned short* x, int rows, int vocab,
                                        float* work, float temperature, int top_k,
                                        float top_p, float min_p, float rng, int* out) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const unsigned short* r = x + (size_t)row * (size_t)vocab;
    float* w = work + (size_t)row * (size_t)vocab;
    const float NEG_INF = __int_as_float(0xff800000);
    float inv_t = (temperature > 0.0f && temperature != 1.0f) ? (1.0f / temperature) : 1.0f;
    float lmax = NEG_INF;
    for (int i = threadIdx.x; i < vocab; i += BLOCK) {
        float v = bf2f(r[i]);
        if (v == v) { v *= inv_t; w[i] = v; if (v > lmax) lmax = v; }
        else { w[i] = NEG_INF; }
    }
    float m = blk_max(lmax);
    finish_row(w, vocab, m, top_k, top_p, min_p, rng, out, row);
}

extern "C" __global__ void sample_f32(const float* x, int rows, int vocab,
                                       float* work, float temperature, int top_k,
                                       float top_p, float min_p, float rng, int* out) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float* r = x + (size_t)row * (size_t)vocab;
    float* w = work + (size_t)row * (size_t)vocab;
    const float NEG_INF = __int_as_float(0xff800000);
    float inv_t = (temperature > 0.0f && temperature != 1.0f) ? (1.0f / temperature) : 1.0f;
    float lmax = NEG_INF;
    for (int i = threadIdx.x; i < vocab; i += BLOCK) {
        float v = r[i];
        if (v == v) { v *= inv_t; w[i] = v; if (v > lmax) lmax = v; }
        else { w[i] = NEG_INF; }
    }
    float m = blk_max(lmax);
    finish_row(w, vocab, m, top_k, top_p, min_p, rng, out, row);
}
"#;

/// A compiled, ready-to-launch on-device sampler bound to device 0's primary
/// context. Cheap to hold; NVRTC compilation happens once at construction.
pub(crate) struct CudaSampler {
    ctx: std::sync::Arc<CudaContext>,
    stream: std::sync::Arc<CudaStream>,
    f_f16: CudaFunction,
    f_bf16: CudaFunction,
    f_f32: CudaFunction,
    /// Non-greedy sampling entry points (temperature/top-k/top-p/min-p +
    /// categorical draw), one per supported logits dtype.
    f_sample_f16: CudaFunction,
    f_sample_bf16: CudaFunction,
    f_sample_f32: CudaFunction,
    /// Reused device scratch holding one `i32` winning index per row. Guarded by
    /// `lock`; grown on demand for wider speculative-verification launches.
    out: Mutex<OutScratch>,
    /// Reused device `f32` scratch of `rows * vocab` slots used by the non-greedy
    /// pipeline to hold the intermediate scaled logits / probabilities. Grown on
    /// demand; guarded by its own `Mutex` (always locked after `out`).
    work: Mutex<WorkScratch>,
}

/// Growable device scratch for the per-row argmax indices.
struct OutScratch {
    ptr: CUdeviceptr,
    /// Capacity in `i32` slots.
    cap: usize,
}

/// Growable device `f32` scratch for the non-greedy sampling pipeline.
struct WorkScratch {
    ptr: CUdeviceptr,
    /// Capacity in `f32` slots.
    cap: usize,
}

// SAFETY: every device operation binds the primary context first and the shared
// `out` scratch is guarded by its `Mutex`, so the handle is safe to move/share
// across threads.
unsafe impl Send for CudaSampler {}
unsafe impl Sync for CudaSampler {}

impl CudaSampler {
    /// Initialise the primary context on device `ordinal`, compile the sampler
    /// module, and allocate the initial result scratch.
    pub(crate) fn new(ordinal: usize) -> Result<Self> {
        let ctx = CudaContext::new(ordinal)
            .map_err(|e| OrtError::Cuda(format!("CudaContext::new({ordinal}): {e:?}")))?;
        let stream = ctx.default_stream();

        // Target the device's own compute capability so the PTX matches.
        let arch = compute_arch(&ctx)?;
        let opts = cudarc::nvrtc::CompileOptions {
            options: vec![format!("--gpu-architecture={arch}")],
            ..Default::default()
        };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(ARGMAX_SRC, opts)
            .map_err(|e| OrtError::Cuda(format!("NVRTC compile argmax: {e:?}")))?;
        let module = ctx
            .load_module(ptx)
            .map_err(|e| OrtError::Cuda(format!("load argmax module: {e:?}")))?;
        let f_f16 = module
            .load_function("argmax_f16")
            .map_err(|e| OrtError::Cuda(format!("load argmax_f16: {e:?}")))?;
        let f_bf16 = module
            .load_function("argmax_bf16")
            .map_err(|e| OrtError::Cuda(format!("load argmax_bf16: {e:?}")))?;
        let f_f32 = module
            .load_function("argmax_f32")
            .map_err(|e| OrtError::Cuda(format!("load argmax_f32: {e:?}")))?;

        // Second module: the non-greedy sampling pipeline.
        let sample_opts = cudarc::nvrtc::CompileOptions {
            options: vec![format!("--gpu-architecture={arch}")],
            ..Default::default()
        };
        let sample_ptx = cudarc::nvrtc::compile_ptx_with_opts(SAMPLE_SRC, sample_opts)
            .map_err(|e| OrtError::Cuda(format!("NVRTC compile sample: {e:?}")))?;
        let sample_module = ctx
            .load_module(sample_ptx)
            .map_err(|e| OrtError::Cuda(format!("load sample module: {e:?}")))?;
        let f_sample_f16 = sample_module
            .load_function("sample_f16")
            .map_err(|e| OrtError::Cuda(format!("load sample_f16: {e:?}")))?;
        let f_sample_bf16 = sample_module
            .load_function("sample_bf16")
            .map_err(|e| OrtError::Cuda(format!("load sample_bf16: {e:?}")))?;
        let f_sample_f32 = sample_module
            .load_function("sample_f32")
            .map_err(|e| OrtError::Cuda(format!("load sample_f32: {e:?}")))?;

        // SAFETY: primary context is current after `CudaContext::new`; we own
        // this allocation and free it in `Drop`.
        let ptr = unsafe { cudarc::driver::result::malloc_sync(4) }
            .map_err(|e| OrtError::Cuda(format!("alloc argmax scratch: {e:?}")))?;
        // SAFETY: same context; freed in `Drop`. Grown on first non-greedy call.
        let work_ptr = unsafe { cudarc::driver::result::malloc_sync(4) }
            .map_err(|e| OrtError::Cuda(format!("alloc sample scratch: {e:?}")))?;

        Ok(Self {
            ctx,
            stream,
            f_f16,
            f_bf16,
            f_f32,
            f_sample_f16,
            f_sample_bf16,
            f_sample_f32,
            out: Mutex::new(OutScratch { ptr, cap: 1 }),
            work: Mutex::new(WorkScratch { ptr: work_ptr, cap: 1 }),
        })
    }
}

impl CudaSampler {
    /// Argmax each of `rows` contiguous `vocab`-element rows in the device buffer
    /// at `ptr_addr` (a device pointer, e.g. from
    /// [`crate::value::Value::data_ptr_addr`]), returning one token id per row.
    /// Synchronizes the context first so all of ORT's just-issued decode work
    /// (which wrote these logits) is visible to the kernel.
    pub(crate) fn argmax_rows(
        &self,
        dtype: DataType,
        ptr_addr: usize,
        rows: usize,
        vocab: usize,
    ) -> Result<Vec<u32>> {
        if rows == 0 {
            return Ok(Vec::new());
        }
        let mut out = self.out.lock().expect("cuda argmax scratch poisoned");
        self.ctx
            .bind_to_thread()
            .map_err(|e| OrtError::Cuda(format!("bind context: {e:?}")))?;
        // ORT's CUDA EP synchronizes its compute stream at the end of each
        // `Run`, so by the time control returns here the logits are fully
        // written and visible. This device-wide wait is therefore normally a
        // no-op guard (see `ctx_sync_enabled`); it can be disabled via
        // `ONNX_GENAI_ARGMAX_CTX_SYNC=0`.
        if ctx_sync_enabled() {
            cudarc::driver::result::ctx::synchronize()
                .map_err(|e| OrtError::Cuda(format!("ctx synchronize: {e:?}")))?;
        }
        out.ensure(rows)?;

        let func = match dtype {
            DataType::Float16 => &self.f_f16,
            DataType::BFloat16 => &self.f_bf16,
            DataType::Float32 => &self.f_f32,
            other => {
                return Err(OrtError::Cuda(format!(
                    "device argmax unsupported logits dtype {other:?}"
                )));
            }
        };
        let x_ptr = ptr_addr as CUdeviceptr;
        let rows_i = i32::try_from(rows)
            .map_err(|_| OrtError::Cuda(format!("row count {rows} exceeds i32")))?;
        let vocab_i = i32::try_from(vocab)
            .map_err(|_| OrtError::Cuda(format!("vocab {vocab} exceeds i32")))?;
        let cfg = LaunchConfig {
            grid_dim: (rows_i as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = self.stream.launch_builder(func);
        builder
            .arg(&x_ptr)
            .arg(&rows_i)
            .arg(&vocab_i)
            .arg(&out.ptr);
        // SAFETY: `func` is the compiled argmax entry; the argument list matches
        // its (const T*, int, int, int*) signature; `x_ptr` is a live device
        // buffer of `rows * vocab` elements and `out.ptr` holds `rows` i32 slots.
        unsafe { builder.launch(cfg) }
            .map_err(|e| OrtError::Cuda(format!("launch argmax: {e:?}")))?;

        let mut idx = vec![0i32; rows];
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(idx.as_mut_ptr().cast::<u8>(), rows * 4)
        };
        // SAFETY: `out.ptr` holds `rows` live i32 slots; `bytes` covers them.
        unsafe { cudarc::driver::result::memcpy_dtoh_sync(bytes, out.ptr) }
            .map_err(|e| OrtError::Cuda(format!("copy argmax result: {e:?}")))?;
        self.stream
            .synchronize()
            .map_err(|e| OrtError::Cuda(format!("stream synchronize: {e:?}")))?;
        Ok(idx.into_iter().map(|v| v as u32).collect())
    }

    /// Non-greedy sampling: apply temperature, top-k, top-p, min-p and a final
    /// inverse-CDF categorical draw to each of `rows` contiguous `vocab`-element
    /// rows at `ptr_addr`, returning the sampled token id per row.
    ///
    /// Every row is sampled independently but with the **same** `rng_value`
    /// (there is one RNG scalar in [`DeviceSampleParams`]). For the usual decode
    /// case `rows == 1` this is exact; for speculative-decode verification
    /// (`rows > 1`) each verified position is still sampled from its own
    /// (correctly filtered/renormalized) distribution, just sharing the single
    /// draw value — a deliberate, documented simplification.
    pub(crate) fn sample_rows(
        &self,
        dtype: DataType,
        ptr_addr: usize,
        rows: usize,
        vocab: usize,
        params: &DeviceSampleParams,
    ) -> Result<Vec<u32>> {
        if rows == 0 {
            return Ok(Vec::new());
        }
        if vocab == 0 {
            return Err(OrtError::Cuda("device sample requires vocab > 0".into()));
        }
        let mut out = self.out.lock().expect("cuda sample out scratch poisoned");
        let mut work = self.work.lock().expect("cuda sample work scratch poisoned");
        self.ctx
            .bind_to_thread()
            .map_err(|e| OrtError::Cuda(format!("bind context: {e:?}")))?;
        if ctx_sync_enabled() {
            cudarc::driver::result::ctx::synchronize()
                .map_err(|e| OrtError::Cuda(format!("ctx synchronize: {e:?}")))?;
        }
        out.ensure(rows)?;
        let work_slots = rows
            .checked_mul(vocab)
            .ok_or_else(|| OrtError::Cuda("sample scratch size overflow".into()))?;
        work.ensure(work_slots)?;

        let func = match dtype {
            DataType::Float16 => &self.f_sample_f16,
            DataType::BFloat16 => &self.f_sample_bf16,
            DataType::Float32 => &self.f_sample_f32,
            other => {
                return Err(OrtError::Cuda(format!(
                    "device sample unsupported logits dtype {other:?}"
                )));
            }
        };
        let x_ptr = ptr_addr as CUdeviceptr;
        let rows_i = i32::try_from(rows)
            .map_err(|_| OrtError::Cuda(format!("row count {rows} exceeds i32")))?;
        let vocab_i = i32::try_from(vocab)
            .map_err(|_| OrtError::Cuda(format!("vocab {vocab} exceeds i32")))?;
        let temperature = params.temperature;
        // Clamp `top_k` into `[0, vocab]` (the kernel treats `>= vocab` as
        // disabled) and into i32 range.
        let top_k_i = i32::try_from(params.top_k.min(vocab)).unwrap_or(i32::MAX);
        let top_p = params.top_p;
        let min_p = params.min_p;
        // Keep the draw strictly < 1.0 so the inverse-CDF always selects a token.
        let rng = if params.rng_value.is_nan() || params.rng_value < 0.0 {
            0.0
        } else if params.rng_value >= 1.0 {
            f32::from_bits(0x3f7f_ffff) // largest f32 < 1.0
        } else {
            params.rng_value
        };
        let cfg = LaunchConfig {
            grid_dim: (rows_i as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = self.stream.launch_builder(func);
        builder
            .arg(&x_ptr)
            .arg(&rows_i)
            .arg(&vocab_i)
            .arg(&work.ptr)
            .arg(&temperature)
            .arg(&top_k_i)
            .arg(&top_p)
            .arg(&min_p)
            .arg(&rng)
            .arg(&out.ptr);
        // SAFETY: `func` is the compiled sample entry; the argument list matches
        // its (const T*, int, int, float*, float, int, float, float, float, int*)
        // signature; `x_ptr` is a live `rows * vocab` device buffer, `work.ptr`
        // holds `rows * vocab` f32 slots and `out.ptr` holds `rows` i32 slots.
        unsafe { builder.launch(cfg) }
            .map_err(|e| OrtError::Cuda(format!("launch sample: {e:?}")))?;

        let mut idx = vec![0i32; rows];
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(idx.as_mut_ptr().cast::<u8>(), rows * 4)
        };
        // SAFETY: `out.ptr` holds `rows` live i32 slots; `bytes` covers them.
        unsafe { cudarc::driver::result::memcpy_dtoh_sync(bytes, out.ptr) }
            .map_err(|e| OrtError::Cuda(format!("copy sample result: {e:?}")))?;
        self.stream
            .synchronize()
            .map_err(|e| OrtError::Cuda(format!("stream synchronize: {e:?}")))?;
        Ok(idx.into_iter().map(|v| v as u32).collect())
    }
}

impl DeviceSampler for CudaSampler {
    fn sample(
        &self,
        dtype: DataType,
        ptr_addr: usize,
        rows: usize,
        vocab: usize,
        params: &DeviceSampleParams,
    ) -> Result<Vec<u32>> {
        // Greedy is exact via argmax and ignores every monotonic filter
        // (temperature/top-k/top-p/min-p never move the maximum).
        if params.greedy {
            return self.argmax_rows(dtype, ptr_addr, rows, vocab);
        }
        // Non-greedy: temperature/top-k/top-p/min-p + categorical draw on-device.
        self.sample_rows(dtype, ptr_addr, rows, vocab, params)
    }

    /// Copy a `len`-element device row of `dtype` at `ptr_addr` into `dst`
    /// (host), for the non-greedy path that still needs the full vocabulary.
    /// Synchronizes the context first so ORT's writes are visible.
    fn copy_row_to_host(
        &self,
        dtype: DataType,
        ptr_addr: usize,
        len: usize,
        dst: &mut [u8],
    ) -> Result<()> {
        let _guard = self.out.lock().expect("cuda argmax scratch poisoned");
        self.ctx
            .bind_to_thread()
            .map_err(|e| OrtError::Cuda(format!("bind context: {e:?}")))?;
        if ctx_sync_enabled() {
            cudarc::driver::result::ctx::synchronize()
                .map_err(|e| OrtError::Cuda(format!("ctx synchronize: {e:?}")))?;
        }
        let want = len
            .checked_mul(dtype_size(dtype)?)
            .ok_or_else(|| OrtError::Cuda("logits byte size overflow".into()))?;
        if dst.len() != want {
            return Err(OrtError::Cuda(format!(
                "device logits copy expected {want} bytes, got {}",
                dst.len()
            )));
        }
        // SAFETY: `ptr_addr` is a live device row of `want` bytes; `dst` matches.
        unsafe { cudarc::driver::result::memcpy_dtoh_sync(dst, ptr_addr as CUdeviceptr) }
            .map_err(|e| OrtError::Cuda(format!("copy logits to host: {e:?}")))?;
        self.stream
            .synchronize()
            .map_err(|e| OrtError::Cuda(format!("stream synchronize: {e:?}")))
    }

    fn name(&self) -> &str {
        "cuda_sampler"
    }
}

impl OutScratch {
    /// Ensure the scratch can hold `rows` i32 indices, reallocating if needed.
    fn ensure(&mut self, rows: usize) -> Result<()> {
        if rows <= self.cap {
            return Ok(());
        }
        // SAFETY: `ptr` came from `malloc_sync`; free before replacing.
        let _ = unsafe { cudarc::driver::result::free_sync(self.ptr) };
        let bytes = rows
            .checked_mul(4)
            .ok_or_else(|| OrtError::Cuda("argmax scratch size overflow".into()))?;
        // SAFETY: primary context is current (caller bound it); we own the result.
        self.ptr = unsafe { cudarc::driver::result::malloc_sync(bytes) }
            .map_err(|e| OrtError::Cuda(format!("grow argmax scratch: {e:?}")))?;
        self.cap = rows;
        Ok(())
    }
}

impl WorkScratch {
    /// Ensure the scratch can hold `slots` f32 values, reallocating if needed.
    fn ensure(&mut self, slots: usize) -> Result<()> {
        if slots <= self.cap {
            return Ok(());
        }
        // SAFETY: `ptr` came from `malloc_sync`; free before replacing.
        let _ = unsafe { cudarc::driver::result::free_sync(self.ptr) };
        let bytes = slots
            .checked_mul(4)
            .ok_or_else(|| OrtError::Cuda("sample scratch size overflow".into()))?;
        // SAFETY: primary context is current (caller bound it); we own the result.
        self.ptr = unsafe { cudarc::driver::result::malloc_sync(bytes) }
            .map_err(|e| OrtError::Cuda(format!("grow sample scratch: {e:?}")))?;
        self.cap = slots;
        Ok(())
    }
}

impl Drop for CudaSampler {
    fn drop(&mut self) {
        // Best-effort free of the scratch; ignore errors during teardown.
        if self.ctx.bind_to_thread().is_ok() {
            if let Ok(out) = self.out.lock() {
                // SAFETY: `out.ptr` came from `malloc_sync` and is freed once here.
                let _ = unsafe { cudarc::driver::result::free_sync(out.ptr) };
            }
            if let Ok(work) = self.work.lock() {
                // SAFETY: `work.ptr` came from `malloc_sync` and is freed once here.
                let _ = unsafe { cudarc::driver::result::free_sync(work.ptr) };
            }
        }
    }
}

/// `compute_XY` string for the device's CUDA compute capability.
fn compute_arch(ctx: &CudaContext) -> Result<String> {
    use cudarc::driver::sys::CUdevice_attribute;
    let major = ctx
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .map_err(|e| OrtError::Cuda(format!("query CC major: {e:?}")))?;
    let minor = ctx
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        .map_err(|e| OrtError::Cuda(format!("query CC minor: {e:?}")))?;
    Ok(format!("compute_{major}{minor}"))
}

fn dtype_size(dtype: DataType) -> Result<usize> {
    Ok(match dtype {
        DataType::Float16 | DataType::BFloat16 => 2,
        DataType::Float32 => 4,
        other => {
            return Err(OrtError::Cuda(format!(
                "device logits unsupported dtype {other:?}"
            )));
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round `x` to an IEEE binary16 bit pattern (round-to-nearest-even on the
    /// retained mantissa; subnormals flushed to zero). Test logits are chosen to
    /// be exactly representable so this simple path preserves ordering.
    fn f32_to_f16_bits(x: f32) -> u16 {
        let bits = x.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
        let mant = bits & 0x007f_ffff;
        if exp <= 0 {
            return sign; // flush subnormal/zero
        } else if exp >= 0x1f {
            return sign | 0x7c00; // inf/overflow
        }
        let m = (mant >> 13) as u16;
        let round_bit = (mant >> 12) & 1;
        let mut h = sign | ((exp as u16) << 10) | m;
        if round_bit == 1 {
            h = h.wrapping_add(1);
        }
        h
    }

    /// Truncate `x` to bfloat16 (the high 16 bits of the f32).
    fn f32_to_bf16_bits(x: f32) -> u16 {
        (x.to_bits() >> 16) as u16
    }

    /// Encode an f32 logits row into the device byte layout for `dtype`.
    fn encode(logits: &[f32], dtype: DataType) -> Vec<u8> {
        match dtype {
            DataType::Float32 => logits.iter().flat_map(|v| v.to_le_bytes()).collect(),
            DataType::Float16 => logits
                .iter()
                .flat_map(|&v| f32_to_f16_bits(v).to_le_bytes())
                .collect(),
            DataType::BFloat16 => logits
                .iter()
                .flat_map(|&v| f32_to_bf16_bits(v).to_le_bytes())
                .collect(),
            other => panic!("unsupported test dtype {other:?}"),
        }
    }

    /// Upload `rows` concatenated logits rows to a fresh device buffer and return
    /// its device pointer address. The buffer is intentionally leaked (freed when
    /// the process exits) to keep the tests simple; they allocate only a handful.
    fn upload(rows: &[Vec<f32>], dtype: DataType) -> usize {
        let mut bytes = Vec::new();
        for row in rows {
            bytes.extend_from_slice(&encode(row, dtype));
        }
        // SAFETY: primary context is current (a `CudaSampler` was constructed on
        // this thread by the caller); we own the allocation.
        let ptr = unsafe { cudarc::driver::result::malloc_sync(bytes.len().max(1)) }
            .expect("malloc device logits");
        // SAFETY: `ptr` holds `bytes.len()` bytes; `bytes` is the matching source.
        unsafe { cudarc::driver::result::memcpy_htod_sync(ptr, &bytes) }
            .expect("htod device logits");
        ptr as usize
    }

    /// Host reference implementing exactly the device pipeline (in f32) so the
    /// two agree token-for-token on well-separated distributions:
    /// temperature -> stable softmax -> top-k -> top-p -> min-p -> inverse-CDF.
    fn host_pipeline(logits: &[f32], params: &DeviceSampleParams) -> u32 {
        let vocab = logits.len();
        let inv_t = if params.temperature > 0.0 && params.temperature != 1.0 {
            1.0 / params.temperature
        } else {
            1.0
        };
        let z: Vec<f32> = logits
            .iter()
            .map(|&l| if l.is_nan() { f32::NEG_INFINITY } else { l * inv_t })
            .collect();
        let m = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        if m == f32::NEG_INFINITY {
            return 0;
        }
        let mut w: Vec<f32> = z
            .iter()
            .map(|&zi| {
                if zi == f32::NEG_INFINITY {
                    0.0
                } else {
                    (zi - m).exp()
                }
            })
            .collect();
        let s: f32 = w.iter().sum();
        if !(s > 0.0) {
            return 0;
        }
        let inv_s = 1.0 / s;
        for wi in w.iter_mut() {
            *wi *= inv_s;
        }
        let p_max = inv_s;

        let minp_thresh = if params.min_p > 0.0 {
            params.min_p * p_max
        } else {
            0.0
        };

        let mut topk_thresh = 0.0f32;
        let top_k = params.top_k.min(vocab);
        if top_k > 0 && top_k < vocab {
            let mut thr = f32::INFINITY;
            for _ in 0..top_k {
                let mut lm = f32::NEG_INFINITY;
                for &p in &w {
                    if p < thr && p > lm {
                        lm = p;
                    }
                }
                thr = lm;
                if thr == f32::NEG_INFINITY {
                    break;
                }
            }
            topk_thresh = if thr == f32::NEG_INFINITY { 0.0 } else { thr };
        }

        let mut topp_thresh = 0.0f32;
        if params.top_p > 0.0 && params.top_p < 1.0 {
            let (mut lo, mut hi) = (0.0f32, p_max);
            for _ in 0..32 {
                let mid = 0.5 * (lo + hi);
                let sm: f32 = w.iter().filter(|&&p| p >= mid).sum();
                if sm >= params.top_p {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            topp_thresh = lo;
        }

        let t = topk_thresh.max(topp_thresh).max(minp_thresh);
        let ssurv: f32 = w.iter().filter(|&&p| p >= t).sum();
        let rng = if params.rng_value.is_nan() || params.rng_value < 0.0 {
            0.0
        } else if params.rng_value >= 1.0 {
            f32::from_bits(0x3f7f_ffff)
        } else {
            params.rng_value
        };
        let target = rng * ssurv;
        let mut acc = 0.0f32;
        for (i, &p) in w.iter().enumerate() {
            if p >= t {
                acc += p;
                if acc > target {
                    return i as u32;
                }
            }
        }
        0
    }

    fn host_argmax(logits: &[f32]) -> u32 {
        let mut best = f32::NEG_INFINITY;
        let mut bidx = 0u32;
        for (i, &v) in logits.iter().enumerate() {
            if v.is_nan() {
                continue;
            }
            if v > best {
                best = v;
                bidx = i as u32;
            }
        }
        bidx
    }

    fn params(temperature: f32, top_k: usize, top_p: f32, min_p: f32, rng: f32) -> DeviceSampleParams {
        DeviceSampleParams {
            temperature,
            top_k,
            top_p,
            min_p,
            greedy: false,
            rng_value: rng,
        }
    }

    /// A varied but well-separated logits row (distinct probabilities with clear
    /// gaps) so ULP-level reduction differences never flip the selected token.
    fn sample_logits() -> Vec<f32> {
        vec![
            0.5, 3.0, 1.0, 5.5, 2.0, 4.25, 0.75, 6.0, 1.5, 3.75, 2.5, 0.25, 4.75, 1.25, 5.0, 2.75,
        ]
    }

    fn new_sampler() -> Option<CudaSampler> {
        match CudaSampler::new(0) {
            Ok(s) => Some(s),
            Err(e) => {
                eprintln!("skipping CUDA sampler test (no device / NVRTC): {e:?}");
                None
            }
        }
    }

    #[test]
    fn greedy_matches_host_argmax_all_dtypes() {
        let Some(sampler) = new_sampler() else { return };
        let logits = sample_logits();
        let vocab = logits.len();
        let expected = host_argmax(&logits);
        for dtype in [DataType::Float32, DataType::Float16, DataType::BFloat16] {
            let ptr = upload(&[logits.clone()], dtype);
            let ids = sampler
                .sample(dtype, ptr, 1, vocab, &DeviceSampleParams::greedy())
                .expect("greedy sample");
            assert_eq!(ids, vec![expected], "greedy dtype {dtype:?}");
        }
    }

    #[test]
    fn categorical_matches_host_pipeline_f32() {
        let Some(sampler) = new_sampler() else { return };
        let logits = sample_logits();
        let vocab = logits.len();
        let ptr = upload(&[logits.clone()], DataType::Float32);
        let combos = [
            params(1.0, 0, 1.0, 0.0, 0.05),
            params(1.0, 0, 1.0, 0.0, 0.5),
            params(1.0, 0, 1.0, 0.0, 0.95),
            params(0.7, 0, 1.0, 0.0, 0.5),
            params(1.5, 0, 1.0, 0.0, 0.5),
            params(1.0, 4, 1.0, 0.0, 0.5),
            params(1.0, 4, 1.0, 0.0, 0.9),
            params(1.0, 0, 0.9, 0.0, 0.5),
            params(1.0, 0, 0.5, 0.0, 0.8),
            params(1.0, 0, 1.0, 0.1, 0.5),
            params(1.0, 0, 1.0, 0.3, 0.99),
            params(0.8, 8, 0.95, 0.05, 0.42),
            params(0.8, 8, 0.95, 0.05, 0.02),
            params(0.8, 8, 0.95, 0.05, 0.77),
        ];
        for p in combos {
            let expected = host_pipeline(&logits, &p);
            let ids = sampler
                .sample(DataType::Float32, ptr, 1, vocab, &p)
                .expect("categorical sample");
            assert_eq!(ids, vec![expected], "params {p:?}");
        }
    }

    #[test]
    fn filters_never_exclude_the_argmax() {
        let Some(sampler) = new_sampler() else { return };
        let logits = sample_logits();
        let vocab = logits.len();
        let argmax = host_argmax(&logits);
        let ptr = upload(&[logits.clone()], DataType::Float32);
        // With rng == 0 the inverse-CDF returns the first surviving index; the
        // argmax must always survive every filter, but it may not be index 0.
        // Instead assert every aggressive filter, when sampled at rng->1, can only
        // ever return a surviving token, and that a top_k=1 / tiny top_p / large
        // min_p collapse the survivor set to exactly the argmax.
        for p in [
            params(1.0, 1, 1.0, 0.0, 0.9),
            params(1.0, 0, 0.01, 0.0, 0.9),
            params(1.0, 0, 1.0, 0.999, 0.9),
        ] {
            let ids = sampler
                .sample(DataType::Float32, ptr, 1, vocab, &p)
                .expect("filtered sample");
            assert_eq!(ids, vec![argmax], "collapsing filter {p:?} must yield argmax");
        }
    }

    #[test]
    fn peaked_distribution_always_returns_argmax() {
        let Some(sampler) = new_sampler() else { return };
        // One logit dominates so far that every other probability underflows to
        // exactly 0.0 in f32 (exp(-120) == 0), leaving a single surviving token;
        // any rng must therefore pick it.
        let mut logits = vec![0.0f32; 32];
        logits[19] = 120.0;
        let vocab = logits.len();
        let ptr = upload(&[logits.clone()], DataType::Float32);
        for &rng in &[0.0f32, 0.01, 0.25, 0.5, 0.75, 0.999999] {
            let p = params(1.0, 0, 1.0, 0.0, rng);
            let ids = sampler
                .sample(DataType::Float32, ptr, 1, vocab, &p)
                .expect("peaked sample");
            assert_eq!(ids, vec![19], "peaked rng {rng}");
        }
    }

    #[test]
    fn multi_row_samples_each_row_with_shared_rng() {
        let Some(sampler) = new_sampler() else { return };
        let row0 = sample_logits();
        let mut row1 = sample_logits();
        row1.reverse();
        let vocab = row0.len();
        let ptr = upload(&[row0.clone(), row1.clone()], DataType::Float32);
        let p = params(0.9, 8, 0.95, 0.02, 0.6);
        let ids = sampler
            .sample(DataType::Float32, ptr, 2, vocab, &p)
            .expect("multi-row sample");
        assert_eq!(ids[0], host_pipeline(&row0, &p), "row 0");
        assert_eq!(ids[1], host_pipeline(&row1, &p), "row 1 (shared rng)");
    }
}
